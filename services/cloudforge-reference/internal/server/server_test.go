package server

import (
	"context"
	"net"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	"cloudforge-reference/internal/config"
	v1 "cloudforge-reference/gen/reference/v1"
)

func TestServer_Integration(t *testing.T) {
	cfg := &config.Config{
		Port:           "0",
		ServiceName:    "cloudforge-reference",
		Version:        "v1.0.0",
		Platform:       "cloudforge",
		Environment:    "test",
		MaxRecvMsgSize: 32 * 1024 * 1024,
		MaxSendMsgSize: 32 * 1024 * 1024,
		MetricsEnabled: false, // Disable standalone HTTP server on port 9090 in unit test to prevent port conflicts
	}

	srv := New(cfg, nil)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on random port: %v", err)
	}

	go func() {
		_ = srv.StartWithListener(lis)
	}()
	defer srv.Stop()

	// Wait briefly for server to start
	time.Sleep(100 * time.Millisecond)

	conn, err := grpc.NewClient(
		lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(cfg.MaxRecvMsgSize),
			grpc.MaxCallSendMsgSize(cfg.MaxSendMsgSize),
		),
	)
	if err != nil {
		t.Fatalf("failed to connect to server: %v", err)
	}
	defer conn.Close()

	// 1. Test Health Check
	healthClient := healthpb.NewHealthClient(conn)
	healthResp, err := healthClient.Check(context.Background(), &healthpb.HealthCheckRequest{})
	if err != nil {
		t.Fatalf("health check failed: %v", err)
	}
	if healthResp.GetStatus() != healthpb.HealthCheckResponse_SERVING {
		t.Errorf("expected health status SERVING, got %v", healthResp.GetStatus())
	}

	// 2. Test Service Health Check
	svcHealthResp, err := healthClient.Check(context.Background(), &healthpb.HealthCheckRequest{
		Service: "cloudforge.reference.v1.ReferenceService",
	})
	if err != nil {
		t.Fatalf("service health check failed: %v", err)
	}
	if svcHealthResp.GetStatus() != healthpb.HealthCheckResponse_SERVING {
		t.Errorf("expected service health status SERVING, got %v", svcHealthResp.GetStatus())
	}

	// 3. Test ReferenceService RPC Success
	refClient := v1.NewReferenceServiceClient(conn)
	statusResp, err := refClient.GetStatus(context.Background(), &v1.StatusRequest{
		Service: "test-caller",
	})
	if err != nil {
		t.Fatalf("GetStatus RPC failed: %v", err)
	}
	if statusResp.GetService() != "test-caller" {
		t.Errorf("expected service test-caller, got %s", statusResp.GetService())
	}
	if statusResp.GetStatus() != "SERVING" {
		t.Errorf("expected status SERVING, got %s", statusResp.GetStatus())
	}
	if statusResp.GetVersion() != "v1.0.0" {
		t.Errorf("expected version v1.0.0, got %s", statusResp.GetVersion())
	}
	if statusResp.GetPlatform() != "cloudforge" {
		t.Errorf("expected platform cloudforge, got %s", statusResp.GetPlatform())
	}

	// 4. Test Invalid Argument handling through interceptors
	tooLongName := strings.Repeat("x", 300)
	_, err = refClient.GetStatus(context.Background(), &v1.StatusRequest{
		Service: tooLongName,
	})
	if err == nil {
		t.Fatalf("expected error for invalid service name, got nil")
	}
	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument error, got %v", st.Code())
	}
}
