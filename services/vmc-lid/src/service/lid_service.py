import time
from typing import Optional
import grpc

from config.settings import Settings
from core.model_registry import ModelRegistry, ModelMetadata
from core.model_runtime import ModelRuntime, AudioInferenceInput, LIDInferenceResult
from core.runtime_factory import ModelRuntimeFactory
from core.audio_processor import AudioProcessingError
from routing.canonical_registry import CanonicalRegistry, get_canonical_registry
from routing.lacr import LACRManager, get_lacr_manager
from telemetry.logging import get_logger
from telemetry.metrics import LIDMetrics

import runtimes  # Auto-registers runtimes with ModelRuntimeFactory
from vmc.lid.v1 import lid_pb2, lid_pb2_grpc

logger = get_logger("lid_service")


class LIDService(lid_pb2_grpc.LIDServiceServicer):
    """
    CloudForge VMC-LID Core Service.
    Coordinates ModelRegistry, ModelRuntime, CanonicalRegistry, LACR, and Telemetry.
    """

    def __init__(
        self,
        settings: Settings,
        registry: Optional[ModelRegistry] = None,
        runtime: Optional[ModelRuntime] = None,
        canonical_registry: Optional[CanonicalRegistry] = None,
        lacr_mgr: Optional[LACRManager] = None,
        metrics: Optional[LIDMetrics] = None
    ):
        self.settings = settings
        self.registry = registry or ModelRegistry(
            registry_path=settings.model.registry_path,
            models_base_dir=settings.model.models_dir
        )
        self.canonical_registry = canonical_registry or get_canonical_registry()
        self.lacr = lacr_mgr or get_lacr_manager()
        self.metrics = metrics or LIDMetrics()

        # Initialize or assign runtime
        if runtime is not None:
            self.runtime = runtime
        else:
            meta = self.registry.get_model(settings.model.active_model_id)
            self.runtime = ModelRuntimeFactory.create(meta, settings)

        # Pre-load model
        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        start = time.time()
        try:
            self.runtime.load()
            load_time = time.time() - start
            self.metrics.record_model_load(
                model_id=self.runtime.metadata.model_id,
                version=self.runtime.metadata.version,
                duration_sec=load_time,
                success=True
            )
            self.metrics.set_serving_status(self.settings.service.name, True)
        except Exception as e:
            self.metrics.record_model_load(
                model_id=self.runtime.metadata.model_id,
                version=self.runtime.metadata.version,
                duration_sec=time.time() - start,
                success=False
            )
            self.metrics.set_serving_status(self.settings.service.name, False)
            logger.error(f"Failed to initialize model runtime: {e}")

    def IdentifyLanguage(
        self,
        request: lid_pb2.IdentifyLanguageRequest,
        context: grpc.ServicerContext
    ) -> lid_pb2.IdentifyLanguageResponse:
        start_time = time.perf_counter()
        req_id = request.request_id or f"req_{int(time.time()*1000)}"
        session_id = request.session_id or ""
        chunk_idx = request.chunk_index if request.chunk_index > 0 else 1

        logger.info(
            f"LID_INFERENCE_STARTED: request_id={req_id} session={session_id} chunk={chunk_idx}",
            extra={
                "event": "LID_INFERENCE_STARTED",
                "request_id": req_id,
                "session_id": session_id,
                "chunk_index": chunk_idx,
                "model_id": self.runtime.metadata.model_id,
                "model_version": self.runtime.metadata.version,
                "service": self.settings.service.name,
            }
        )

        # 1. Validate Audio Input
        if not request.audio_content or len(request.audio_content) == 0:
            err_msg = "Audio content payload is empty"
            logger.warning(f"LID_INFERENCE_FAILED: {err_msg}, request_id={req_id}")
            self.metrics.record_error(self.runtime.metadata.model_id, "EMPTY_AUDIO")
            return lid_pb2.IdentifyLanguageResponse(
                language="UNKNOWN",
                language_name="Unknown",
                confidence=0.0,
                language_group=self.canonical_registry.default_group,
                target_vm=self.canonical_registry.default_vm,
                status=lid_pb2.LID_STATUS_INVALID_AUDIO,
                model_id=self.runtime.metadata.model_id,
                model_version=self.runtime.metadata.version,
                request_id=req_id,
                inference_duration_ms=0.0,
                probabilities={}
            )

        # 2. Execute Inference via Generic ModelRuntime
        try:
            inp = AudioInferenceInput(
                audio_data=request.audio_content,
                sample_rate=request.sample_rate or 16000,
                channels=request.channels or 1,
                audio_format=request.audio_format or "pcm_s16le",
                request_id=req_id
            )
            inf_result = self.runtime.infer(inp)
        except AudioProcessingError as ape:
            logger.warning(f"LID_INFERENCE_FAILED: audio decoding error '{str(ape)}', request_id={req_id}")
            self.metrics.record_error(self.runtime.metadata.model_id, "AUDIO_PROCESSING_ERROR")
            return lid_pb2.IdentifyLanguageResponse(
                language="UNKNOWN",
                language_name="Unknown",
                confidence=0.0,
                language_group=self.canonical_registry.default_group,
                target_vm=self.canonical_registry.default_vm,
                status=lid_pb2.LID_STATUS_INVALID_AUDIO,
                model_id=self.runtime.metadata.model_id,
                model_version=self.runtime.metadata.version,
                request_id=req_id,
                inference_duration_ms=0.0,
                probabilities={}
            )
        except Exception as e:
            logger.error(f"LID_INFERENCE_FAILED: runtime error '{str(e)}', request_id={req_id}")
            self.metrics.record_error(self.runtime.metadata.model_id, "RUNTIME_INFERENCE_ERROR")
            return lid_pb2.IdentifyLanguageResponse(
                language="UNKNOWN",
                language_name="Unknown",
                confidence=0.0,
                language_group=self.canonical_registry.default_group,
                target_vm=self.canonical_registry.default_vm,
                status=lid_pb2.LID_STATUS_INFERENCE_ERROR,
                model_id=self.runtime.metadata.model_id,
                model_version=self.runtime.metadata.version,
                request_id=req_id,
                inference_duration_ms=0.0,
                probabilities={}
            )

        # 3. Resolve Language Group & Target VM from Canonical Registry
        extra_status = inf_result.extra.get("status")
        predicted_lang = inf_result.language

        if extra_status == "MODEL_HEAD_NOT_AVAILABLE":
            status = lid_pb2.LID_STATUS_MODEL_HEAD_NOT_AVAILABLE
            group_id = self.canonical_registry.default_group
            target_vm = self.canonical_registry.default_vm
            language_name = "Unknown"
        elif extra_status == "MODEL_NOT_AVAILABLE":
            status = lid_pb2.LID_STATUS_MODEL_NOT_AVAILABLE
            group_id = self.canonical_registry.default_group
            target_vm = self.canonical_registry.default_vm
            language_name = "Unknown"
        else:
            lang_info = self.canonical_registry.get_language_info(predicted_lang)

            if lang_info is None:
                group_id = self.canonical_registry.default_group
                target_vm = self.canonical_registry.default_vm
                language_name = predicted_lang
                status = lid_pb2.LID_STATUS_UNSUPPORTED
            else:
                group_id = lang_info.group
                target_vm = lang_info.vm
                language_name = lang_info.name

                # 4. Evaluate Confidence Policy
                threshold = self.settings.model.confidence_threshold
                is_low_conf = inf_result.confidence < threshold

                if is_low_conf:
                    status = lid_pb2.LID_STATUS_LOW_CONFIDENCE
                    logger.warning(
                        f"LID_LOW_CONFIDENCE: predicted={inf_result.language} confidence={inf_result.confidence:.3f} < threshold={threshold:.3f}",
                        extra={
                            "event": "LID_LOW_CONFIDENCE",
                            "request_id": req_id,
                            "model_id": self.runtime.metadata.model_id,
                            "model_version": self.runtime.metadata.version,
                            "language": inf_result.language,
                            "confidence": inf_result.confidence,
                        }
                    )
                else:
                    status = lid_pb2.LID_STATUS_SUCCESS

        # 5. LACR Session Tracking
        effective_lang, effective_group, effective_vm, has_switched = self.lacr.update_session(
            session_id=session_id,
            chunk_index=chunk_idx,
            predicted_language=predicted_lang,
            predicted_group=group_id,
            predicted_vm=target_vm,
            confidence=inf_result.confidence
        )

        total_duration_sec = time.perf_counter() - start_time
        total_duration_ms = total_duration_sec * 1000.0

        # 6. Telemetry & Metrics Recording
        self.metrics.record_inference(
            service=self.settings.service.name,
            model_id=self.runtime.metadata.model_id,
            version=self.runtime.metadata.version,
            duration_sec=total_duration_sec,
            status=lid_pb2.LIDStatus.Name(status),
            language=effective_lang,
            is_low_confidence=(status == lid_pb2.LID_STATUS_LOW_CONFIDENCE)
        )

        logger.info(
            f"LID_INFERENCE_COMPLETED: lang={effective_lang} conf={inf_result.confidence:.3f} group={effective_group} vm={effective_vm} ({total_duration_ms:.2f}ms)",
            extra={
                "event": "LID_INFERENCE_COMPLETED",
                "request_id": req_id,
                "model_id": self.runtime.metadata.model_id,
                "model_version": self.runtime.metadata.version,
                "language": effective_lang,
                "confidence": inf_result.confidence,
                "language_group": effective_group,
                "target_vm": effective_vm,
                "duration_ms": total_duration_ms,
                "status": lid_pb2.LIDStatus.Name(status)
            }
        )

        return lid_pb2.IdentifyLanguageResponse(
            language=effective_lang,
            language_name=self.canonical_registry.get_language_name(effective_lang, language_name),
            confidence=inf_result.confidence,
            language_group=effective_group,
            target_vm=effective_vm,
            status=status,
            model_id=self.runtime.metadata.model_id,
            model_version=self.runtime.metadata.version,
            request_id=req_id,
            inference_duration_ms=inf_result.inference_duration_ms,
            probabilities=inf_result.probabilities
        )

    def GetModelInfo(
        self,
        request: lid_pb2.ModelInfoRequest,
        context: grpc.ServicerContext
    ) -> lid_pb2.ModelInfoResponse:
        target_model_id = request.model_id or self.runtime.metadata.model_id
        try:
            meta = self.registry.get_model(target_model_id)
        except Exception:
            meta = self.runtime.metadata

        return lid_pb2.ModelInfoResponse(
            model_id=meta.model_id,
            model_name=meta.model_name,
            provider=meta.provider,
            framework=meta.framework,
            version=meta.version,
            parameters=meta.parameters,
            sample_rate=meta.sample_rate,
            channels=meta.channels,
            supported_languages=meta.supported_languages,
            status=meta.status,
            artifact_size_mb=meta.artifact_size_mb
        )

    def GetSupportedLanguages(
        self,
        request: lid_pb2.GetSupportedLanguagesRequest,
        context: grpc.ServicerContext
    ) -> lid_pb2.GetSupportedLanguagesResponse:
        active_langs = set(self.runtime.metadata.supported_languages)
        canonical_list = self.canonical_registry.list_canonical_languages()

        filter_group = request.group.upper() if request.group else ""

        proto_langs = []
        for l in canonical_list:
            if filter_group and l.group != filter_group:
                continue
            proto_langs.append(
                lid_pb2.LanguageMetadata(
                    language_code=l.code,
                    language_name=l.name,
                    native_name=l.native_name,
                    family=l.family,
                    group=l.group,
                    target_vm=l.vm,
                    is_supported_by_active_model=(l.code in active_langs)
                )
            )

        return lid_pb2.GetSupportedLanguagesResponse(
            canonical_languages=proto_langs,
            active_model_languages=list(self.runtime.metadata.supported_languages),
            total_canonical=len(canonical_list),
            total_active=len(active_langs)
        )

    def ResolveLanguageGroup(
        self,
        request: lid_pb2.ResolveLanguageGroupRequest,
        context: grpc.ServicerContext
    ) -> lid_pb2.ResolveLanguageGroupResponse:
        info = self.canonical_registry.get_language_info(request.language)
        if info:
            return lid_pb2.ResolveLanguageGroupResponse(
                language=info.code,
                language_name=info.name,
                group=info.group,
                target_vm=info.vm,
                target_service=info.service_name,
                target_port=info.port,
                is_canonical=True
            )

        return lid_pb2.ResolveLanguageGroupResponse(
            language=request.language,
            language_name="Unknown",
            group=self.canonical_registry.default_group,
            target_vm=self.canonical_registry.default_vm,
            target_service="",
            target_port=0,
            is_canonical=False
        )
