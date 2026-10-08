# CloudForge Protobuf Generation Script
# Handles Windows Device Guard explicit plugin paths, Python grpc_tools, and Linux environments

param (
    [string]$Service = "all"
)

$ErrorActionPreference = "Stop"

$GoPlugin = "$env:USERPROFILE\go\bin\protoc-gen-go.exe"
$GrpcPlugin = "$env:USERPROFILE\go\bin\protoc-gen-go-grpc.exe"

if (-not (Test-Path $GoPlugin)) {
    $GoPlugin = "protoc-gen-go"
}
if (-not (Test-Path $GrpcPlugin)) {
    $GrpcPlugin = "protoc-gen-go-grpc"
}

function Generate-Reference {
    Write-Host "Generating Protobuf stubs for service: reference" -ForegroundColor Cyan
    $ServiceDir = "C:\CloudForge\services\cloudforge-reference"
    Push-Location $ServiceDir
    try {
        protoc --plugin="protoc-gen-go=$GoPlugin" `
               --plugin="protoc-gen-go-grpc=$GrpcPlugin" `
               --proto_path=proto `
               --go_out=. `
               --go-grpc_out=. `
               proto/reference.proto
        Write-Host "Successfully generated stubs in $ServiceDir\gen" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}

function Generate-LID {
    Write-Host "Generating Protobuf stubs for service: vmc-lid" -ForegroundColor Cyan
    $RootProtoDir = "C:\CloudForge\proto"
    $LidOutDir = "C:\CloudForge\services\vmc-lid\src\gen"
    
    New-Item -ItemType Directory -Force -Path "$LidOutDir\vmc\lid\v1" | Out-Null
    New-Item -ItemType File -Force -Path "$LidOutDir\__init__.py" | Out-Null
    New-Item -ItemType File -Force -Path "$LidOutDir\vmc\__init__.py" | Out-Null
    New-Item -ItemType File -Force -Path "$LidOutDir\vmc\lid\__init__.py" | Out-Null
    New-Item -ItemType File -Force -Path "$LidOutDir\vmc\lid\v1\__init__.py" | Out-Null

    python -m grpc_tools.protoc `
           -I="$RootProtoDir" `
           --python_out="$LidOutDir" `
           --grpc_python_out="$LidOutDir" `
           "$RootProtoDir\vmc\lid\v1\lid.proto"

    # Also generate Go stubs if Go protoc plugin exists
    $LidGoOutDir = "C:\CloudForge\services\vmc-lid\gen"
    if (Test-Path $GoPlugin) {
        New-Item -ItemType Directory -Force -Path $LidGoOutDir | Out-Null
        protoc --plugin="protoc-gen-go=$GoPlugin" `
               --plugin="protoc-gen-go-grpc=$GrpcPlugin" `
               -I="$RootProtoDir" `
               --go_out="$LidGoOutDir" `
               --go-grpc_out="$LidGoOutDir" `
               "$RootProtoDir\vmc\lid\v1\lid.proto"
    }

    Write-Host "Successfully generated Python and Go stubs for vmc-lid" -ForegroundColor Green
}

function Generate-Detector {
    Write-Host "Generating Protobuf stubs for detector services" -ForegroundColor Cyan
    $RootProtoDir = "C:\CloudForge\proto"
    $ProtoFile = "$RootProtoDir\vmc\detector\v1\detector.proto"
    
    if (-not (Test-Path $ProtoFile)) {
        Write-Warning "Detector proto not found at $ProtoFile"
        return
    }

    $DetectorServices = @("vmc-south", "vmc-north", "vmc-east", "vmc-frontier", "vmc-west", "vmc-english")
    foreach ($srv in $DetectorServices) {
        $OutDir = "C:\CloudForge\services\$srv\src\gen"
        New-Item -ItemType Directory -Force -Path "$OutDir\vmc\detector\v1" | Out-Null
        New-Item -ItemType File -Force -Path "$OutDir\__init__.py" | Out-Null
        New-Item -ItemType File -Force -Path "$OutDir\vmc\__init__.py" | Out-Null
        New-Item -ItemType File -Force -Path "$OutDir\vmc\detector\__init__.py" | Out-Null
        New-Item -ItemType File -Force -Path "$OutDir\vmc\detector\v1\__init__.py" | Out-Null

        python -m grpc_tools.protoc `
               -I="$RootProtoDir" `
               --python_out="$OutDir" `
               --grpc_python_out="$OutDir" `
               $ProtoFile
    }
    Write-Host "Successfully generated detector stubs for all 6 detector clusters" -ForegroundColor Green
}

switch ($Service) {
    "reference" {
        Generate-Reference
    }
    "lid" {
        Generate-LID
    }
    "vmc-lid" {
        Generate-LID
    }
    "detector" {
        Generate-Detector
    }
    "all" {
        Generate-Reference
        Generate-LID
        Generate-Detector
    }
    Default {
        Write-Error "Unknown service: $Service"
    }
}
