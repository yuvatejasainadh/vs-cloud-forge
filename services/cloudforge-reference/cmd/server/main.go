package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"cloudforge-reference/internal/config"
	"cloudforge-reference/internal/server"
)

func main() {
	// Initialize JSON structured logger
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	cfg := config.Load()

	logger.Info("Initializing CloudForge Reference Service",
		"service", cfg.ServiceName,
		"version", cfg.Version,
		"platform", cfg.Platform,
		"port", cfg.Port,
		"metrics_port", cfg.MetricsPort,
		"metrics_enabled", cfg.MetricsEnabled,
		"max_recv_mb", cfg.MaxRecvMsgSize/(1024*1024),
		"max_send_mb", cfg.MaxSendMsgSize/(1024*1024),
		"environment", cfg.Environment,
	)

	srv := server.New(cfg, logger)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if err := srv.Run(ctx); err != nil {
		logger.Error("Server stopped with error", "error", err)
		os.Exit(1)
	}

	logger.Info("CloudForge Reference Service stopped successfully")
}
