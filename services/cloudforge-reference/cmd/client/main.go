package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	v1 "cloudforge-reference/gen/reference/v1"
)

func main() {
	target := flag.String("addr", "localhost:50051", "Target gRPC server address (host:port)")
	serviceName := flag.String("service", "cloudforge-reference", "Service name to request status for")
	checkHealth := flag.Bool("health", false, "Perform standard gRPC health check instead of GetStatus")
	timeout := flag.Duration("timeout", 5*time.Second, "Request timeout")
	flag.Parse()

	conn, err := grpc.NewClient(*target, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to connect to %s: %v\n", *target, err)
		os.Exit(1)
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	if *checkHealth {
		healthClient := healthpb.NewHealthClient(conn)
		resp, err := healthClient.Check(ctx, &healthpb.HealthCheckRequest{
			Service: *serviceName,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "Health check failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Health Status: %s\n", resp.GetStatus().String())
		return
	}

	refClient := v1.NewReferenceServiceClient(conn)
	resp, err := refClient.GetStatus(ctx, &v1.StatusRequest{
		Service: *serviceName,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "GetStatus RPC failed: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("=== Reference Service Response ===")
	fmt.Printf("Service:  %s\n", resp.GetService())
	fmt.Printf("Status:   %s\n", resp.GetStatus())
	fmt.Printf("Version:  %s\n", resp.GetVersion())
	fmt.Printf("Platform: %s\n", resp.GetPlatform())
}
