import os
import hashlib
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
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

    def __init__(self, base_dirs: Optional[List[str]] = None):
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
        return None


class HuggingFaceArtifactSource(ArtifactSource):
    """
    Resolves and downloads model artifacts from Hugging Face Hub using pinned revisions.
    Maintains deterministic local cache under models/huggingface/<repo>/<revision>/.
    """

    def __init__(
        self,
        repository_id: str = "facebook/wav2vec2-xls-r-300m",
        revision: str = "1a640f32ac3e39899438a2931f9924c02f080a54",
        cache_base_dir: Optional[str] = None,
        auto_download: bool = False
    ):
        self.repository_id = repository_id
        self.revision = revision
        self.auto_download = auto_download
        if cache_base_dir:
            self.cache_dir = cache_base_dir
        else:
            # Deterministic location: models/huggingface/<org>/<model>/<revision>/
            parts = repository_id.replace("/", os.sep)
            self.cache_dir = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "..",
                    "models", "huggingface", parts, revision
                )
            )

    def resolve_artifact(
        self,
        model_id: str,
        version: str,
        filename: str = "model.onnx",
        allow_patterns: Optional[List[str]] = None
    ) -> Optional[str]:
        candidates = [filename]
        if filename == "model.onnx":
            candidates.extend(["pytorch_model.bin", "model_quantized.onnx", "model.safetensors"])

        for candidate in candidates:
            local_target = os.path.join(self.cache_dir, candidate)
            if os.path.exists(local_target) and os.path.isfile(local_target):
                logger.info(f"Resolved artifact '{candidate}' from local HuggingFace cache: {local_target}")
                return local_target

        if not self.auto_download:
            return None

        # Attempt download via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(self.cache_dir, exist_ok=True)
            for candidate in candidates:
                try:
                    logger.info(
                        f"Attempting download of '{candidate}' from HF repo '{self.repository_id}' (rev: {self.revision})..."
                    )
                    downloaded_path = hf_hub_download(
                        repo_id=self.repository_id,
                        filename=candidate,
                        revision=self.revision,
                        local_dir=self.cache_dir,
                        local_dir_use_symlinks=False
                    )
                    if os.path.exists(downloaded_path):
                        return downloaded_path
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Could not download artifacts from HuggingFace ({self.repository_id}@{self.revision}): {e}")

        return None


class ArtifactResolver:
    """
    CloudForge Model Artifact Resolver and Integrity Verifier.
    Ensures only bit-exact, verified model weights are hydrated and loaded into runtimes.
    """

    def __init__(self, sources: Optional[List[ArtifactSource]] = None):
        self.sources = sources or [
            LocalFilesystemSource(),
            HuggingFaceArtifactSource()
        ]

    def resolve_and_verify(
        self,
        model_id: str,
        version: str,
        filename: str = "model.onnx",
        expected_sha256: Optional[str] = None
    ) -> str:
        artifact_path = None
        for source in self.sources:
            path = source.resolve_artifact(model_id, version, filename)
            if path and os.path.exists(path):
                artifact_path = path
                break

        if not artifact_path or not os.path.exists(artifact_path):
            raise ArtifactNotFoundError(
                f"Model artifact '{filename}' for model '{model_id}' (v{version}) not found in any source."
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
