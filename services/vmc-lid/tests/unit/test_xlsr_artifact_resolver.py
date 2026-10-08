import os
import pytest
import tempfile
from core.artifact_resolver import (
    ArtifactResolver,
    HuggingFaceArtifactSource,
    LocalFilesystemSource,
    ArtifactNotFoundError,
    ChecksumMismatchError,
)


class TestXLSRArtifactResolver:

    def test_huggingface_resolution(self):
        """Verifies HuggingFaceArtifactSource initializes with correct repo and revision."""
        source = HuggingFaceArtifactSource(
            repository_id="facebook/wav2vec2-xls-r-300m",
            revision="1a640f32ac3e39899438a2931f9924c02f080a54"
        )
        assert source.repository_id == "facebook/wav2vec2-xls-r-300m"
        assert source.revision == "1a640f32ac3e39899438a2931f9924c02f080a54"

    def test_revision_pinning(self):
        """Ensures that revision is strictly pinned and not unpinned main."""
        source = HuggingFaceArtifactSource(
            repository_id="facebook/wav2vec2-xls-r-300m",
            revision="1a640f32ac3e39899438a2931f9924c02f080a54"
        )
        assert source.revision != "main"
        assert len(source.revision) == 40  # Full SHA-1 commit hash

    def test_cache_path(self):
        """Verifies deterministic cache directory path calculation."""
        source = HuggingFaceArtifactSource(
            repository_id="facebook/wav2vec2-xls-r-300m",
            revision="1a640f32ac3e39899438a2931f9924c02f080a54"
        )
        assert "models" in source.cache_dir
        assert "huggingface" in source.cache_dir
        assert "1a640f32ac3e39899438a2931f9924c02f080a54" in source.cache_dir

    def test_missing_artifact(self):
        """Verifies ArtifactNotFoundError is raised when artifact is not present in cache/storage."""
        resolver = ArtifactResolver(sources=[LocalFilesystemSource(base_dirs=["/nonexistent/dir"])])
        with pytest.raises(ArtifactNotFoundError):
            resolver.resolve_and_verify(
                model_id="nonexistent-model",
                version="v1.0.0",
                filename="model.onnx"
            )

    def test_checksum_verification(self):
        """Verifies SHA-256 calculation and mismatch detection."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"cloudforge-xlsr-test-payload")
            temp_path = f.name

        try:
            actual_sha = ArtifactResolver.compute_sha256(temp_path)
            assert len(actual_sha) == 64

            # Valid checksum passes
            resolver = ArtifactResolver()
            assert resolver.compute_sha256(temp_path) == actual_sha

            # Invalid checksum raises ChecksumMismatchError when resolved
            class MockSource(LocalFilesystemSource):
                def resolve_artifact(self, model_id, version, filename):
                    return temp_path

            mock_resolver = ArtifactResolver(sources=[MockSource()])
            with pytest.raises(ChecksumMismatchError):
                mock_resolver.resolve_and_verify(
                    model_id="test-model",
                    version="v1.0",
                    expected_sha256="0000000000000000000000000000000000000000000000000000000000000000"
                )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
