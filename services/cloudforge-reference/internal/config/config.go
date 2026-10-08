package config

import (
	"os"
	"strconv"
)

const (
	// DefaultMaxMsgSizeMB is the default maximum gRPC message size in megabytes (32 MB).
	// This limit accommodates raw uncompressed audio chunks, multi-channel arrays,
	// and batch inference payloads for VMC services while guarding against unbounded memory allocation.
	DefaultMaxMsgSizeMB = 32
	DefaultMaxMsgSizeBytes = DefaultMaxMsgSizeMB * 1024 * 1024
)

// Config holds the application configuration.
type Config struct {
	Port           string
	ServiceName    string
	Version        string
	Platform       string
	Environment    string
	MaxRecvMsgSize int
	MaxSendMsgSize int
	MetricsPort    string
	MetricsEnabled bool
}

// Load reads configuration from environment variables with sensible defaults.
func Load() *Config {
	port := getEnv("PORT", "50051")
	if grpcPort := os.Getenv("GRPC_PORT"); grpcPort != "" {
		port = grpcPort
	}

	maxRecv := parseMsgSize("GRPC_MAX_RECV_MSG_SIZE_MB", "GRPC_MAX_RECV_MSG_SIZE_BYTES", DefaultMaxMsgSizeBytes)
	maxSend := parseMsgSize("GRPC_MAX_SEND_MSG_SIZE_MB", "GRPC_MAX_SEND_MSG_SIZE_BYTES", DefaultMaxMsgSizeBytes)

	metricsEnabled := true
	if val := os.Getenv("METRICS_ENABLED"); val != "" {
		if parsed, err := strconv.ParseBool(val); err == nil {
			metricsEnabled = parsed
		}
	}

	return &Config{
		Port:           port,
		ServiceName:    getEnv("SERVICE_NAME", "cloudforge-reference"),
		Version:        getEnv("SERVICE_VERSION", "v1.0.0"),
		Platform:       getEnv("PLATFORM_NAME", "cloudforge"),
		Environment:    getEnv("ENVIRONMENT", "local"),
		MaxRecvMsgSize: maxRecv,
		MaxSendMsgSize: maxSend,
		MetricsPort:    getEnv("METRICS_PORT", "9090"),
		MetricsEnabled: metricsEnabled,
	}
}

// Address returns the gRPC listen address string (e.g. ":50051").
func (c *Config) Address() string {
	return ":" + c.Port
}

// MetricsAddress returns the Prometheus metrics listen address string (e.g. ":9090").
func (c *Config) MetricsAddress() string {
	return ":" + c.MetricsPort
}

func getEnv(key, defaultVal string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return defaultVal
}

func parseMsgSize(mbKey, bytesKey string, defaultBytes int) int {
	if mbVal := os.Getenv(mbKey); mbVal != "" {
		if mb, err := strconv.Atoi(mbVal); err == nil && mb > 0 {
			return mb * 1024 * 1024
		}
	}
	if bytesVal := os.Getenv(bytesKey); bytesVal != "" {
		if bytes, err := strconv.Atoi(bytesVal); err == nil && bytes > 0 {
			return bytes
		}
	}
	return defaultBytes
}
