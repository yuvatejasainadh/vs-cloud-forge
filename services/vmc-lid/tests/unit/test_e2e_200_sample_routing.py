import pytest
import os
import sys
import json
import random

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "gen")))

from tools.evaluate_e2e_200_sample_routing import (
    discover_and_sample_dataset,
    MULTILINGUAL_DATASET_DIR,
    DEEPFAKE_DATASET_DIR,
    RANDOM_SEED,
    SAMPLES_PER_DATASET
)
from routing.canonical_registry import get_canonical_registry


class TestE2E200SampleRouting:

    def test_deterministic_sampling_reproducibility(self):
        """Verify that seed 42 yields bit-exact identical file selections across multiple runs."""
        sample_a = discover_and_sample_dataset(MULTILINGUAL_DATASET_DIR, sample_count=100, seed=42)
        sample_b = discover_and_sample_dataset(MULTILINGUAL_DATASET_DIR, sample_count=100, seed=42)
        assert len(sample_a) == 100
        assert sample_a == sample_b

        sample_df_a = discover_and_sample_dataset(DEEPFAKE_DATASET_DIR, sample_count=100, seed=42)
        sample_df_b = discover_and_sample_dataset(DEEPFAKE_DATASET_DIR, sample_count=100, seed=42)
        assert len(sample_df_a) == 100
        assert sample_df_a == sample_df_b

    def test_e2e_routing_summary_report_invariants(self):
        """Verify the generated summary report satisfies all audit constraints."""
        summary_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "reports", "e2e_200_sample_routing_summary.json"))
        if not os.path.exists(summary_path):
            pytest.skip("E2E summary report not found.")

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        # Assert total sample counts
        assert summary["sample_configuration"]["total_samples"] == 200
        assert summary["sample_configuration"]["multilingual_dataset_samples"] == 100
        assert summary["sample_configuration"]["deepfake_dataset_samples"] == 100

        # Assert routing success
        assert summary["routing_results"]["successfully_routed"] == 200
        assert summary["routing_results"]["routing_success_rate_percent"] == 100.0
        assert summary["routing_results"]["lid_unavailable"] == 0
        assert summary["routing_results"]["unsupported_language"] == 0
        assert summary["routing_results"]["routing_errors"] == 0

        # Assert governance: detector inference strictly bypassed
        assert summary["inference_execution_governance"]["detector_inference_executed"] == 0
        assert summary["inference_execution_governance"]["aasist_l_inference_executed"] == 0

        # Assert final status
        assert summary["final_status"] == "E2E ROUTING VERIFIED"

    def test_e2e_detailed_results_record_integrity(self):
        """Verify each of the 200 detailed records has full metadata and correct contract mapping."""
        detailed_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "reports", "e2e_200_sample_routing_results.json"))
        if not os.path.exists(detailed_path):
            pytest.skip("E2E detailed results report not found.")

        with open(detailed_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        assert len(records) == 200
        can_reg = get_canonical_registry()

        for rec in records:
            assert rec["LID_status"] == "SUCCESS"
            assert rec["preprocessing_status"] == "SUCCESS"
            assert rec["inference_executed"] is False
            assert rec["error"] is None

            lang = rec["detected_language"]
            lang_info = can_reg.get_language_info(lang)
            assert lang_info is not None

            assert rec["semantic_group"] == lang_info.group
            assert rec["target_vm"] == lang_info.vm

            if lang == "en-IN":
                assert rec["target_vm"] == "VM7_ENGLISH"
                assert rec["detector_target"] == "AASIST-L"
                assert rec["final_status"] == "ROUTED_TO_AASIST_L"
            else:
                assert rec["detector_target"] is None
                assert rec["final_status"] == "ROUTED"
