package metrics

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"cloudforge-reference/internal/config"
)

func TestMetrics_InterceptorAndRegistry(t *testing.T) {
	cfg := &config.Config{
		ServiceName: "cloudforge-reference",
		Version:     "v1.0.0",
		Platform:    "cloudforge",
		Environment: "test",
		MetricsPort: "9090",
	}

	m := New(cfg, nil)
	interceptor := m.UnaryServerInterceptor()

	info := &grpc.UnaryServerInfo{
		FullMethod: "/cloudforge.reference.v1.ReferenceService/GetStatus",
	}

	// 1. Test Successful RPC invocation
	handlerSuccess := func(ctx context.Context, req any) (any, error) {
		return "ok-resp", nil
	}
	resp, err := interceptor(context.Background(), "req", info, handlerSuccess)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp != "ok-resp" {
		t.Errorf("expected resp ok-resp, got %v", resp)
	}

	// 2. Test Failed RPC invocation
	handlerError := func(ctx context.Context, req any) (any, error) {
		return nil, status.Error(codes.InvalidArgument, "invalid argument")
	}
	_, err = interceptor(context.Background(), "req", info, handlerError)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}

	// 3. Test Serving Status Change
	m.SetServingStatus(false)
	m.SetServingStatus(true)

	// 4. Test HTTP Metrics Handler Output
	httpServer := m.NewHTTPServer()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()
	httpServer.Handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 from /metrics, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "cloudforge_grpc_requests_total") {
		t.Errorf("expected body to contain cloudforge_grpc_requests_total")
	}
	if !strings.Contains(body, "cloudforge_service_info") {
		t.Errorf("expected body to contain cloudforge_service_info")
	}
	if !strings.Contains(body, "cloudforge_service_health_status") {
		t.Errorf("expected body to contain cloudforge_service_health_status")
	}

	// 5. Test Healthz Handler
	reqHealth := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	wHealth := httptest.NewRecorder()
	httpServer.Handler.ServeHTTP(wHealth, reqHealth)

	if wHealth.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 from /healthz, got %d", wHealth.Code)
	}
	if wHealth.Body.String() != "OK" {
		t.Errorf("expected body OK, got %s", wHealth.Body.String())
	}
}
