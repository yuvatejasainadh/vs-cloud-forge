from typing import Optional
from config.settings import Settings
from core.runtime import AASISTLRuntime
from core.artifact_resolver import ArtifactResolver
from telemetry.logging import get_logger

logger = get_logger("runtime_factory")


class AASISTLRuntimeFactory:
    """
    CloudForge Runtime Factory for instantiating AASIST-L deepfake detection runtimes.
    """

    @classmethod
    def create(
        cls,
        settings: Settings,
        resolver: Optional[ArtifactResolver] = None
    ) -> AASISTLRuntime:
        model_id = settings.detector.model_id
        version = settings.detector.model_version
        threshold = settings.detector.threshold.value
        execution_provider = settings.runtime.execution_provider
        artifact_filename = settings.detector.artifact_filename
        expected_sha256 = settings.detector.expected_sha256

        logger.info(
            f"Instantiating CloudForge AASIST-L runtime for model '{model_id}' "
            f"(version: {version}, execution_provider: {execution_provider})"
        )

        return AASISTLRuntime(
            model_id=model_id,
            version=version,
            artifact_filename=artifact_filename,
            expected_sha256=expected_sha256,
            threshold=threshold,
            execution_provider=execution_provider,
            resolver=resolver
        )
