package config

import (
	"os"
	"testing"
)

func TestConfigLoadDefaults(t *testing.T) {
	os.Unsetenv("PORT")
	os.Unsetenv("GRPC_PORT")
	os.Unsetenv("SERVICE_NAME")
	os.Unsetenv("SERVICE_VERSION")
	os.Unsetenv("PLATFORM_NAME")
	os.Unsetenv("ENVIRONMENT")
	os.Unsetenv("GRPC_MAX_RECV_MSG_SIZE_MB")
	os.Unsetenv("GRPC_MAX_RECV_MSG_SIZE_BYTES")
	os.Unsetenv("GRPC_MAX_SEND_MSG_SIZE_MB")
	os.Unsetenv("GRPC_MAX_SEND_MSG_SIZE_BYTES")
	os.Unsetenv("METRICS_PORT")
	os.Unsetenv("METRICS_ENABLED")

	cfg := Load()

	if cfg.Port != "50051" {
		t.Errorf("expected default Port 50051, got %s", cfg.Port)
	}
	if cfg.ServiceName != "cloudforge-reference" {
		t.Errorf("expected default ServiceName cloudforge-reference, got %s", cfg.ServiceName)
	}
	if cfg.Version != "v1.0.0" {
		t.Errorf("expected default Version v1.0.0, got %s", cfg.Version)
	}
	if cfg.Platform != "cloudforge" {
		t.Errorf("expected default Platform cloudforge, got %s", cfg.Platform)
	}
	if cfg.Environment != "local" {
		t.Errorf("expected default Environment local, got %s", cfg.Environment)
	}
	if cfg.MaxRecvMsgSize != 32*1024*1024 {
		t.Errorf("expected default MaxRecvMsgSize 33554432 (32MB), got %d", cfg.MaxRecvMsgSize)
	}
	if cfg.MaxSendMsgSize != 32*1024*1024 {
		t.Errorf("expected default MaxSendMsgSize 33554432 (32MB), got %d", cfg.MaxSendMsgSize)
	}
	if cfg.MetricsPort != "9090" {
		t.Errorf("expected default MetricsPort 9090, got %s", cfg.MetricsPort)
	}
	if !cfg.MetricsEnabled {
		t.Errorf("expected default MetricsEnabled true, got false")
	}
	if cfg.Address() != ":50051" {
		t.Errorf("expected Address :50051, got %s", cfg.Address())
	}
	if cfg.MetricsAddress() != ":9090" {
		t.Errorf("expected MetricsAddress :9090, got %s", cfg.MetricsAddress())
	}
}

func TestConfigLoadCustomEnv(t *testing.T) {
	t.Setenv("PORT", "9090")
	t.Setenv("SERVICE_NAME", "custom-service")
	t.Setenv("SERVICE_VERSION", "v2.1.0")
	t.Setenv("PLATFORM_NAME", "vmc")
	t.Setenv("ENVIRONMENT", "staging")
	t.Setenv("GRPC_MAX_RECV_MSG_SIZE_MB", "64")
	t.Setenv("GRPC_MAX_SEND_MSG_SIZE_MB", "48")
	t.Setenv("METRICS_PORT", "9100")
	t.Setenv("METRICS_ENABLED", "false")

	cfg := Load()

	if cfg.Port != "9090" {
		t.Errorf("expected Port 9090, got %s", cfg.Port)
	}
	if cfg.ServiceName != "custom-service" {
		t.Errorf("expected ServiceName custom-service, got %s", cfg.ServiceName)
	}
	if cfg.Version != "v2.1.0" {
		t.Errorf("expected Version v2.1.0, got %s", cfg.Version)
	}
	if cfg.Platform != "vmc" {
		t.Errorf("expected Platform vmc, got %s", cfg.Platform)
	}
	if cfg.Environment != "staging" {
		t.Errorf("expected Environment staging, got %s", cfg.Environment)
	}
	if cfg.MaxRecvMsgSize != 64*1024*1024 {
		t.Errorf("expected MaxRecvMsgSize 67108864 (64MB), got %d", cfg.MaxRecvMsgSize)
	}
	if cfg.MaxSendMsgSize != 48*1024*1024 {
		t.Errorf("expected MaxSendMsgSize 50331648 (48MB), got %d", cfg.MaxSendMsgSize)
	}
	if cfg.MetricsPort != "9100" {
		t.Errorf("expected MetricsPort 9100, got %s", cfg.MetricsPort)
	}
	if cfg.MetricsEnabled {
		t.Errorf("expected MetricsEnabled false, got true")
	}
}

func TestConfigLoadCustomBytesEnv(t *testing.T) {
	t.Setenv("GRPC_MAX_RECV_MSG_SIZE_BYTES", "10485760") // 10MB in bytes
	t.Setenv("GRPC_MAX_SEND_MSG_SIZE_BYTES", "20971520") // 20MB in bytes

	cfg := Load()

	if cfg.MaxRecvMsgSize != 10485760 {
		t.Errorf("expected MaxRecvMsgSize 10485760, got %d", cfg.MaxRecvMsgSize)
	}
	if cfg.MaxSendMsgSize != 20971520 {
		t.Errorf("expected MaxSendMsgSize 20971520, got %d", cfg.MaxSendMsgSize)
	}
}
