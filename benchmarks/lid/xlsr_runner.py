import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LID_TOOL_DIR = os.path.join(BASE_DIR, "services", "vmc-lid", "tools")
if LID_TOOL_DIR not in sys.path:
    sys.path.insert(0, LID_TOOL_DIR)

from benchmark_xlsr import run_xlsr_benchmark

if __name__ == "__main__":
    run_xlsr_benchmark()
