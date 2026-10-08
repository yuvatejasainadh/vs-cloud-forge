package server

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	"cloudforge-reference/internal/config"
	"cloudforge-reference/internal/metrics"
	"cloudforge-reference/internal/service"
	v1 "cloudforge-reference/gen/reference/v1"
)

// Server wraps the gRPC server, Prometheus metrics exporter, and lifecycle managers.
type Server struct {
	cfg           *config.Config
	logger        *slog.Logger
	grpcServer    *grpc.Server
	healthServer  *health.Server
	refService    *service.ReferenceService
	metrics       *metrics.Metrics
	metricsServer *http.Server
	listener      net.Listener
	mu            sync.Mutex
	running       bool
}

// New creates and configures a new Server instance.
func New(cfg *config.Config, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}

	serverLogger := logger.With("component", "Server")
	metricsCollector := metrics.New(cfg, logger)

	// Chain standard production unary interceptors:
	// 1. Recovery Interceptor (catches panics)
	// 2. Metrics Interceptor (measures request counts, latencies, status codes)
	// 3. Structured Logging Interceptor (JSON log outputs)
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(cfg.MaxRecvMsgSize),
		grpc.MaxSendMsgSize(cfg.MaxSendMsgSize),
		grpc.ChainUnaryInterceptor(
			recoveryInterceptor(serverLogger),
			metricsCollector.UnaryServerInterceptor(),
			loggingInterceptor(serverLogger),
		),
	)

	healthServer := health.NewServer()
	refService := service.NewReferenceService(cfg, logger)

	// Register Reference service
	v1.RegisterReferenceServiceServer(grpcServer, refService)

	// Register Health service (used by Kubernetes liveness/readiness probes)
	healthpb.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthServer.SetServingStatus(cfg.ServiceName, healthpb.HealthCheckResponse_SERVING)
	healthServer.SetServingStatus("cloudforge.reference.v1.ReferenceService", healthpb.HealthCheckResponse_SERVING)

	// Register Reflection service (allows tools like grpcurl to discover services)
	reflection.Register(grpcServer)

	var metricsServer *http.Server
	if cfg.MetricsEnabled {
		metricsServer = metricsCollector.NewHTTPServer()
	}

	return &Server{
		cfg:           cfg,
		logger:        serverLogger,
		grpcServer:    grpcServer,
		healthServer:  healthServer,
		refService:    refService,
		metrics:       metricsCollector,
		metricsServer: metricsServer,
	}
}

// Start starts the gRPC server and metrics server on their configured addresses.
func (s *Server) Start() error {
	s.mu.Lock()
	lis, err := net.Listen("tcp", s.cfg.Address())
	if err != nil {
		s.mu.Unlock()
		return fmt.Errorf("failed to listen on %s: %w", s.cfg.Address(), err)
	}
	s.listener = lis
	s.running = true
	s.mu.Unlock()

	s.logger.Info("Starting gRPC server",
		"address", s.cfg.Address(),
		"service", s.cfg.ServiceName,
		"version", s.cfg.Version,
		"platform", s.cfg.Platform,
		"environment", s.cfg.Environment,
		"max_recv_mb", s.cfg.MaxRecvMsgSize/(1024*1024),
		"max_send_mb", s.cfg.MaxSendMsgSize/(1024*1024),
	)

	if s.metricsServer != nil {
		go func() {
			s.logger.Info("Starting Prometheus metrics server", "address", s.cfg.MetricsAddress())
			if err := s.metricsServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				s.logger.Error("Metrics server failed", "error", err)
			}
		}()
	}

	if err := s.grpcServer.Serve(lis); err != nil && err != grpc.ErrServerStopped {
		return fmt.Errorf("gRPC server error: %w", err)
	}
	return nil
}

// StartWithListener starts the gRPC server using a provided listener (useful for tests).
func (s *Server) StartWithListener(lis net.Listener) error {
	s.mu.Lock()
	s.listener = lis
	s.running = true
	s.mu.Unlock()

	s.logger.Info("Starting gRPC server on custom listener", "address", lis.Addr().String())

	if err := s.grpcServer.Serve(lis); err != nil && err != grpc.ErrServerStopped {
		return fmt.Errorf("gRPC server error: %w", err)
	}
	return nil
}

// Stop gracefully stops the gRPC and metrics servers with a default 5-second timeout.
func (s *Server) Stop() {
	s.StopWithTimeout(5 * time.Second)
}

// StopWithTimeout gracefully stops the gRPC and HTTP servers with an explicit deadline.
func (s *Server) StopWithTimeout(timeout time.Duration) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.running {
		return
	}

	s.logger.Info("Stopping gRPC and metrics servers gracefully...", "timeout", timeout.String())

	// Update health statuses immediately
	s.healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	s.healthServer.SetServingStatus(s.cfg.ServiceName, healthpb.HealthCheckResponse_NOT_SERVING)
	s.healthServer.SetServingStatus("cloudforge.reference.v1.ReferenceService", healthpb.HealthCheckResponse_NOT_SERVING)
	s.metrics.SetServingStatus(false)

	// Shutdown HTTP metrics server
	if s.metricsServer != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		_ = s.metricsServer.Shutdown(shutdownCtx)
	}

	// Drain and stop gRPC server
	stopped := make(chan struct{})
	go func() {
		s.grpcServer.GracefulStop()
		close(stopped)
	}()

	select {
	case <-stopped:
		s.logger.Info("gRPC server stopped cleanly")
	case <-time.After(timeout):
		s.logger.Warn("Graceful shutdown timed out, forcing stop")
		s.grpcServer.Stop()
	}

	s.running = false
}

// Run starts the server and gracefully stops when context is canceled.
func (s *Server) Run(ctx context.Context) error {
	errCh := make(chan error, 1)

	go func() {
		if err := s.Start(); err != nil {
			errCh <- err
		}
	}()

	select {
	case <-ctx.Done():
		s.logger.Info("Context canceled, initiating graceful shutdown")
		s.Stop()
		return nil
	case err := <-errCh:
		return err
	}
}

// GRPCServer returns the underlying *grpc.Server.
func (s *Server) GRPCServer() *grpc.Server {
	return s.grpcServer
}

// Metrics returns the Prometheus metrics collector instance.
func (s *Server) Metrics() *metrics.Metrics {
	return s.metrics
}

// loggingInterceptor logs request duration, method, and gRPC status code.
func loggingInterceptor(logger *slog.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		duration := time.Since(start)
		st, _ := status.FromError(err)

		if err != nil {
			logger.ErrorContext(ctx, "gRPC request completed with error",
				"method", info.FullMethod,
				"duration_ms", duration.Milliseconds(),
				"code", st.Code().String(),
				"error", err.Error(),
			)
		} else {
			logger.InfoContext(ctx, "gRPC request completed successfully",
				"method", info.FullMethod,
				"duration_ms", duration.Milliseconds(),
				"code", st.Code().String(),
			)
		}
		return resp, err
	}
}

// recoveryInterceptor catches unhandled panics and returns an Internal error code.
func recoveryInterceptor(logger *slog.Logger) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
		defer func() {
			if r := recover(); r != nil {
				logger.ErrorContext(ctx, "panic recovered in gRPC handler",
					"method", info.FullMethod,
					"panic", fmt.Sprintf("%v", r),
				)
				err = status.Errorf(codes.Internal, "internal server error")
			}
		}()
		return handler(ctx, req)
	}
}
