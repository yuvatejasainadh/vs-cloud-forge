from core.runtime_factory import ModelRuntimeFactory
from runtimes.open_vakgyata_runtime import OpenVakgyataRuntime
from runtimes.xlsr_runtime import XLSRRuntime

# Register default ONNX runtime and model specific runtimes
ModelRuntimeFactory.register("onnx", OpenVakgyataRuntime)
ModelRuntimeFactory.register("open-vakgyata", OpenVakgyataRuntime)
ModelRuntimeFactory.register("facebook-wav2vec2-xls-r-300m", XLSRRuntime)
ModelRuntimeFactory.register("xls-r", XLSRRuntime)
ModelRuntimeFactory.register("wav2vec2-xls-r", XLSRRuntime)
ModelRuntimeFactory.register("transformers", XLSRRuntime)
