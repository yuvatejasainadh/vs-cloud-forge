import os
import sys
import tempfile
import hashlib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from core.artifact_resolver import (
    ArtifactResolver,
    LocalFilesystemSource,
    ArtifactNotFoundError,
    ChecksumMismatchError
)


class TestArtifactResolver:
    def test_missing_artifact_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = LocalFilesystemSource(base_dirs=[tmpdir])
            resolver = ArtifactResolver(source)

            with pytest.raises(ArtifactNotFoundError):
                resolver.resolve_and_verify("nonexistent-model", "v1.0.0", "model.onnx")

    def test_checksum_verification_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "voiceshield-aasist-english", "v1.0.0")
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, "model.onnx")

            content = b"fake_onnx_model_content_for_checksum_test"
            with open(model_file, "wb") as f:
                f.write(content)

            expected_sha256 = hashlib.sha256(content).hexdigest()

            source = LocalFilesystemSource(base_dirs=[tmpdir])
            resolver = ArtifactResolver(source)

            resolved_path = resolver.resolve_and_verify(
                model_id="voiceshield-aasist-english",
                version="v1.0.0",
                filename="model.onnx",
                expected_sha256=expected_sha256
            )

            assert resolved_path == os.path.abspath(model_file)

    def test_checksum_verification_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "voiceshield-aasist-english", "v1.0.0")
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, "model.onnx")

            with open(model_file, "wb") as f:
                f.write(b"corrupted_content")

            source = LocalFilesystemSource(base_dirs=[tmpdir])
            resolver = ArtifactResolver(source)

            with pytest.raises(ChecksumMismatchError):
                resolver.resolve_and_verify(
                    model_id="voiceshield-aasist-english",
                    version="v1.0.0",
                    filename="model.onnx",
                    expected_sha256="0000000000000000000000000000000000000000000000000000000000000000"
                )
