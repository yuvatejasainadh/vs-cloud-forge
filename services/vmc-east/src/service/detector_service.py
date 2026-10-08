import time
import grpc
from vmc.detector.v1 import detector_pb2, detector_pb2_grpc


class DetectorService(detector_pb2_grpc.DetectorServiceServicer):
    """
    Scaffolded detector service for East Indian Group (Eastern Indo-Aryan).
    Target VM: VM4_EAST | Group: EAST | Port: 50054
    """

    def __init__(self):
        self.vm_id = "VM4_EAST"
        self.group = "EAST"
        self.supported_languages = ['bn-IN', 'as-IN', 'or-IN', 'bpy-IN']
        self.model_id = "voiceshield-aasist-east"
        self.model_type = "MULTILINGUAL_AASIST_L"
        self.model_version = "v1.0.0-scaffold"

    def DetectSpoof(
        self,
        request: detector_pb2.DetectSpoofRequest,
        context: grpc.ServicerContext
    ) -> detector_pb2.DetectSpoofResponse:
        start_time = time.perf_counter()
        req_id = request.request_id or f"req_{int(time.time()*1000)}"

        if not request.audio_content or len(request.audio_content) == 0:
            return detector_pb2.DetectSpoofResponse(
                spoof_score=0.0,
                bonafide_score=0.0,
                status=detector_pb2.DETECTION_STATUS_INVALID_AUDIO,
                language_group=self.group,
                target_vm=self.vm_id,
                model_id=self.model_id,
                model_version=self.model_version,
                request_id=req_id,
                inference_duration_ms=0.0
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return detector_pb2.DetectSpoofResponse(
            spoof_score=0.05,
            bonafide_score=0.95,
            status=detector_pb2.DETECTION_STATUS_GENUINE,
            language_group=self.group,
            target_vm=self.vm_id,
            model_id=self.model_id,
            model_version=self.model_version,
            request_id=req_id,
            inference_duration_ms=duration_ms
        )

    def GetDetectorInfo(
        self,
        request: detector_pb2.DetectorInfoRequest,
        context: grpc.ServicerContext
    ) -> detector_pb2.DetectorInfoResponse:
        return detector_pb2.DetectorInfoResponse(
            model_id=self.model_id,
            model_name="VoiceShield AASIST-L (EAST)",
            target_vm=self.vm_id,
            group=self.group,
            model_type=self.model_type,
            supported_languages=self.supported_languages,
            status="SCAFFOLDED"
        )
