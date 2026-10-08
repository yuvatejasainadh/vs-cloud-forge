package metrics

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"

	"cloudforge-reference/internal/config"
)

// Metrics holds all platform metrics collectors and Prometheus registry.
type Metrics struct {
	Registry          *prometheus.Registry
	RequestsTotal     *prometheus.CounterVec
	RequestDuration   *prometheus.HistogramVec
	ErrorsTotal       *prometheus.CounterVec
	ServiceInfo       *prometheus.GaugeVec
	ServiceHealth     *prometheus.GaugeVec
	InferenceDuration *prometheus.HistogramVec
	cfg               *config.Config
	logger            *slog.Logger
}

// New initializes all standard Prometheus metrics collectors.
func New(cfg *config.Config, logger *slog.Logger) *Metrics {
	if logger == nil {
		logger = slog.Default()
	}

	reg := prometheus.NewRegistry()

	// Default Go and Process collectors
	reg.MustRegister(prometheus.NewGoCollector())
	reg.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))

	requestsTotal := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "cloudforge",
			Subsystem: "grpc",
			Name:      "requests_total",
			Help:      "Total number of gRPC requests processed partitioned by service, method, and response status code.",
		},
		[]string{"service", "method", "code"},
	)

	requestDuration := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "cloudforge",
			Subsystem: "grpc",
			Name:      "request_duration_seconds",
			Help:      "Latency distribution of gRPC requests in seconds.",
			Buckets: []float64{
				0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
			},
		},
		[]string{"service", "method"},
	)

	errorsTotal := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "cloudforge",
			Subsystem: "grpc",
			Name:      "errors_total",
			Help:      "Total number of gRPC errors partitioned by service, method, and code.",
		},
		[]string{"service", "method", "code"},
	)

	serviceInfo := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "cloudforge",
			Subsystem: "service",
			Name:      "info",
			Help:      "Static service metadata gauge with constant value 1.",
		},
		[]string{"service", "version", "platform", "environment"},
	)

	serviceHealth := prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "cloudforge",
			Subsystem: "service",
			Name:      "health_status",
			Help:      "Current health status gauge (1 = SERVING, 0 = NOT_SERVING).",
		},
		[]string{"service"},
	)

	inferenceDuration := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "cloudforge",
			Subsystem: "ml",
			Name:      "inference_duration_seconds",
			Help:      "Latency distribution of ML model inference operations in seconds.",
			Buckets: []float64{
				0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
			},
		},
		[]string{"service", "model", "version"},
	)

	reg.MustRegister(requestsTotal, requestDuration, errorsTotal, serviceInfo, serviceHealth, inferenceDuration)

	// Set initial service metadata
	serviceInfo.WithLabelValues(cfg.ServiceName, cfg.Version, cfg.Platform, cfg.Environment).Set(1)
	serviceHealth.WithLabelValues(cfg.ServiceName).Set(1)

	return &Metrics{
		Registry:          reg,
		RequestsTotal:     requestsTotal,
		RequestDuration:   requestDuration,
		ErrorsTotal:       errorsTotal,
		ServiceInfo:       serviceInfo,
		ServiceHealth:     serviceHealth,
		InferenceDuration: inferenceDuration,
		cfg:               cfg,
		logger:            logger.With("component", "Metrics"),
	}
}

// UnaryServerInterceptor returns a gRPC interceptor that records request counts, latencies, and error codes.
func (m *Metrics) UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		start := time.Now()
		resp, err := handler(ctx, req)
		duration := time.Since(start).Seconds()

		st, _ := status.FromError(err)
		codeStr := st.Code().String()

		m.RequestsTotal.WithLabelValues(m.cfg.ServiceName, info.FullMethod, codeStr).Inc()
		m.RequestDuration.WithLabelValues(m.cfg.ServiceName, info.FullMethod).Observe(duration)

		if err != nil {
			m.ErrorsTotal.WithLabelValues(m.cfg.ServiceName, info.FullMethod, codeStr).Inc()
		}

		return resp, err
	}
}

// SetServingStatus updates the service health status metric.
func (m *Metrics) SetServingStatus(serving bool) {
	val := float64(0)
	if serving {
		val = 1
	}
	m.ServiceHealth.WithLabelValues(m.cfg.ServiceName).Set(val)
}

// NewHTTPServer creates an http.Server configured with the Prometheus /metrics endpoint.
func (m *Metrics) NewHTTPServer() *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(m.Registry, promhttp.HandlerOpts{
		EnableOpenMetrics: true,
	}))
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})

	return &http.Server{
		Addr:         m.cfg.MetricsAddress(),
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}
}
