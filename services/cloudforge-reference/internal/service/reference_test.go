package service

import (
	"context"
	"strings"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"cloudforge-reference/internal/config"
	v1 "cloudforge-reference/gen/reference/v1"
)

func TestReferenceService_GetStatus(t *testing.T) {
	cfg := &config.Config{
		Port:        "50051",
		ServiceName: "cloudforge-reference",
		Version:     "v1.0.0",
		Platform:    "cloudforge",
		Environment: "test",
	}

	svc := NewReferenceService(cfg, nil)

	t.Run("default service name when empty", func(t *testing.T) {
		resp, err := svc.GetStatus(context.Background(), &v1.StatusRequest{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.GetService() != "cloudforge-reference" {
			t.Errorf("expected service 'cloudforge-reference', got %s", resp.GetService())
		}
		if resp.GetStatus() != "SERVING" {
			t.Errorf("expected status 'SERVING', got %s", resp.GetStatus())
		}
		if resp.GetVersion() != "v1.0.0" {
			t.Errorf("expected version 'v1.0.0', got %s", resp.GetVersion())
		}
		if resp.GetPlatform() != "cloudforge" {
			t.Errorf("expected platform 'cloudforge', got %s", resp.GetPlatform())
		}
	})

	t.Run("custom service name in request", func(t *testing.T) {
		resp, err := svc.GetStatus(context.Background(), &v1.StatusRequest{
			Service: "custom-query-service",
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if resp.GetService() != "custom-query-service" {
			t.Errorf("expected service 'custom-query-service', got %s", resp.GetService())
		}
		if resp.GetStatus() != "SERVING" {
			t.Errorf("expected status 'SERVING', got %s", resp.GetStatus())
		}
	})

	t.Run("invalid service name length exceeds limit", func(t *testing.T) {
		tooLongName := strings.Repeat("a", 300)
		_, err := svc.GetStatus(context.Background(), &v1.StatusRequest{
			Service: tooLongName,
		})
		if err == nil {
			t.Fatalf("expected error for overly long service name, got nil")
		}
		st, ok := status.FromError(err)
		if !ok || st.Code() != codes.InvalidArgument {
			t.Errorf("expected InvalidArgument error code, got %v", st.Code())
		}
	})
}
