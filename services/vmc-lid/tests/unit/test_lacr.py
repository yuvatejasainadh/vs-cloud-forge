import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from routing.lacr import LACRManager


class TestLACR:
    def test_checkpoint_intervals(self):
        lacr = LACRManager(checkpoint_interval=5)
        session_id = "sess_test_1"

        # Chunk 1 requires LID evaluation
        assert lacr.should_evaluate_lid(session_id, chunk_index=1) is True

        # Update session with initial prediction
        lacr.update_session(session_id, 1, "hi-IN", "NORTH", "VM3_NORTH", 0.92)

        # Chunks 2, 3, 4, 5 do NOT require evaluation
        assert lacr.should_evaluate_lid(session_id, chunk_index=2) is False
        assert lacr.should_evaluate_lid(session_id, chunk_index=3) is False
        assert lacr.should_evaluate_lid(session_id, chunk_index=4) is False
        assert lacr.should_evaluate_lid(session_id, chunk_index=5) is False

        # Chunk 6 is a checkpoint -> requires evaluation
        assert lacr.should_evaluate_lid(session_id, chunk_index=6) is True

    def test_mid_call_code_switching(self):
        lacr = LACRManager(checkpoint_interval=5, switch_confidence_threshold=0.80)
        session_id = "sess_switch_1"

        # Initial chunk in Telugu
        lang, group, vm, switched = lacr.update_session(session_id, 1, "te-IN", "SOUTH", "VM2_SOUTH", 0.95)
        assert lang == "te-IN"
        assert group == "SOUTH"
        assert vm == "VM2_SOUTH"
        assert switched is False

        # Low confidence divergence in Hindi -> should retain Telugu
        lang, group, vm, switched = lacr.update_session(session_id, 6, "hi-IN", "NORTH", "VM3_NORTH", 0.60)
        assert lang == "te-IN"
        assert vm == "VM2_SOUTH"
        assert switched is False

        # High confidence switch to Hindi -> triggers language switch
        lang, group, vm, switched = lacr.update_session(session_id, 11, "hi-IN", "NORTH", "VM3_NORTH", 0.91)
        assert lang == "hi-IN"
        assert group == "NORTH"
        assert vm == "VM3_NORTH"
        assert switched is True
