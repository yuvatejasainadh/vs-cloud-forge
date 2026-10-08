package service

import (
	"context"
	"log/slog"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"cloudforge-reference/internal/config"
	v1 "cloudforge-reference/gen/reference/v1"
)

const maxServiceNameLength = 256

// ReferenceService implements the v1.ReferenceServiceServer interface.
type ReferenceService struct {
	v1.UnimplementedReferenceServiceServer
	cfg    *config.Config
	logger *slog.Logger
}

// NewReferenceService creates a new ReferenceService instance.
func NewReferenceService(cfg *config.Config, logger *slog.Logger) *ReferenceService {
	if logger == nil {
		logger = slog.Default()
	}
	return &ReferenceService{
		cfg:    cfg,
		logger: logger.With("component", "ReferenceService"),
	}
}

// GetStatus returns the current status, version, and platform metadata of the service.
func (s *ReferenceService) GetStatus(ctx context.Context, req *v1.StatusRequest) (*v1.StatusResponse, error) {
	requestedService := s.cfg.ServiceName
	if req != nil && req.GetService() != "" {
		if len(req.GetService()) > maxServiceNameLength {
			s.logger.WarnContext(ctx, "Invalid service name length in request", "service", req.GetService())
			return nil, status.Errorf(codes.InvalidArgument, "service name exceeds maximum allowed length of %d characters", maxServiceNameLength)
		}
		requestedService = req.GetService()
	}

	s.logger.InfoContext(ctx, "Handling GetStatus request",
		"requested_service", requestedService,
		"configured_service", s.cfg.ServiceName,
		"version", s.cfg.Version,
		"platform", s.cfg.Platform,
	)

	return &v1.StatusResponse{
		Service:  requestedService,
		Status:   "SERVING",
		Version:  s.cfg.Version,
		Platform: s.cfg.Platform,
	}, nil
}
