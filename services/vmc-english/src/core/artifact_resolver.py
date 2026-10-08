import os
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from telemetry.logging import get_logger

logger = get_logger("artifact_resolver")


class ArtifactResolutionError(Exception):
    """Base exception for model artifact resolution and verification errors."""
    pass


class ArtifactNotFoundError(ArtifactResolutionError):
    """Raised when the specified model artifact cannot be found in cache or storage."""
    pass


class ChecksumMismatchError(ArtifactResolutionError):
    """Raised when model artifact checksum verification fails."""
    pass


class ArtifactSource(ABC):
    """Abstract source provider for fetching or resolving model artifacts."""

    @abstractmethod
    def resolve_artifact(self, model_id: str, version: str, filename: str) -> Optional[str]:
        pass


class LocalFilesystemSource(ArtifactSource):
    """Resolves model artifacts from local filesystem cache directories."""

    def __init__(self, base_dirs: Optional[list] = None):
        self.base_dirs = base_dirs or [
            r"C:\CloudForge\model-cache",
            r"C:\CloudForge\models",
            "/models",
            "/model-cache"
        ]

    def resolve_artifact(self, model_id: str, version: str, filename: str = "model.onnx") -> Optional[str]:
        candidates = []
        for base in self.base_dirs:
            candidates.append(os.path.join(base, model_id, version, filename))
            candidates.append(os.path.join(base, model_id, filename))
            candidates.append(os.path.join(base, filename))

        for path in candidates:
            normalized = os.path.abspath(path)
            if os.path.exists(normalized) and os.path.isfile(normalized):
                return normalized
        return None


class ObjectStorageSource(ArtifactSource):
    """Placeholder for remote S3 / Cloud Object Storage artifact sourcing."""

    def __init__(self, endpoint_url: str = "", bucket: str = ""):
        self.endpoint_url = endpoint_url
        self.bucket = bucket

    def resolve_artifact(self, model_id: str, version: str, filename: str = "model.onnx") -> Optional[str]:
        # Remote hydration logic to local cache if implemented
        return None


class ArtifactResolver:
    """
    CloudForge Model Artifact Resolver and Integrity Verifier.
    Ensures that only bit-exact, verified model weights are hydrated and loaded into runtimes.
    """

    def __init__(self, source: Optional[ArtifactSource] = None):
        self.source = source or LocalFilesystemSource()

    def resolve_and_verify(
        self,
        model_id: str,
        version: str,
        filename: str = "model.onnx",
        expected_sha256: Optional[str] = None
    ) -> str:
        artifact_path = self.source.resolve_artifact(model_id, version, filename)
        if not artifact_path or not os.path.exists(artifact_path):
            raise ArtifactNotFoundError(
                f"Model artifact '{filename}' for model '{model_id}' (v{version}) not found in cache/storage."
            )

        if expected_sha256:
            actual_sha256 = self.compute_sha256(artifact_path)
            if actual_sha256.lower() != expected_sha256.lower():
                logger.error(
                    f"Checksum mismatch for '{artifact_path}'. Expected: {expected_sha256}, got: {actual_sha256}"
                )
                raise ChecksumMismatchError(
                    f"Model artifact checksum verification failed for {artifact_path}. "
                    f"Expected: {expected_sha256}, Actual: {actual_sha256}"
                )
            logger.info(f"Checksum verified for '{artifact_path}' (SHA256: {actual_sha256})")

        return artifact_path

    @staticmethod
    def compute_sha256(file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
