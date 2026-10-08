import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
import json
from routing.canonical_registry import get_canonical_registry
from routing.language_group_resolver import LanguageGroupResolver


SELECTED_12_LANGUAGES = {
    "te-IN": {"name": "Telugu", "group": "SOUTH", "vm": "VM2_SOUTH", "service": "vmc-south", "port": 50052},
    "ta-IN": {"name": "Tamil", "group": "SOUTH", "vm": "VM2_SOUTH", "service": "vmc-south", "port": 50052},
    "ml-IN": {"name": "Malayalam", "group": "SOUTH", "vm": "VM2_SOUTH", "service": "vmc-south", "port": 50052},
    "kn-IN": {"name": "Kannada", "group": "SOUTH", "vm": "VM2_SOUTH", "service": "vmc-south", "port": 50052},
    "hi-IN": {"name": "Hindi", "group": "NORTH", "vm": "VM3_NORTH", "service": "vmc-north", "port": 50053},
    "mr-IN": {"name": "Marathi", "group": "NORTH", "vm": "VM3_NORTH", "service": "vmc-north", "port": 50053},
    "sa-IN": {"name": "Sanskrit", "group": "NORTH", "vm": "VM3_NORTH", "service": "vmc-north", "port": 50053},
    "bn-IN": {"name": "Bengali", "group": "EAST", "vm": "VM4_EAST", "service": "vmc-east", "port": 50054},
    "as-IN": {"name": "Assamese", "group": "EAST", "vm": "VM4_EAST", "service": "vmc-east", "port": 50054},
    "or-IN": {"name": "Odia", "group": "EAST", "vm": "VM4_EAST", "service": "vmc-east", "port": 50054},
    "gu-IN": {"name": "Gujarati", "group": "WEST", "vm": "VM6_WEST", "service": "vmc-west", "port": 50056},
    "en-IN": {"name": "English", "group": "ENGLISH", "vm": "VM7_ENGLISH", "service": "vmc-english", "port": 50057},
}

EXCLUDED_DATASET_LANGUAGES = ["Manipuri", "Bodo", "Dogri", "Nepali"]


@pytest.fixture
def resolver():
    registry = get_canonical_registry()
    return LanguageGroupResolver(registry=registry)


class Test12LanguageRoutingEvaluation:

    @pytest.mark.parametrize("code,expected", list(SELECTED_12_LANGUAGES.items()))
    def test_all_12_canonical_routing(self, resolver, code, expected):
        """Verify each of the 12 selected canonical languages routes to its expected VM and group."""
        res = resolver.resolve(code)
        assert res is not None
        assert res.group == expected["group"]
        assert res.vm == expected["vm"]
        assert res.service_name == expected["service"]
        assert res.port == expected["port"]

    def test_excluded_languages_do_not_resolve_to_active_vms(self, resolver):
        """Verify non-target / non-canonical languages fall back cleanly."""
        for ex in ["brx-IN", "doi-IN", "nep-IN", "mni-IN", "unknown-IN", ""]:
            group = resolver.resolve_group(ex)
            vm = resolver.resolve_vm(ex)
            assert group == "UNKNOWN_GROUP"
            assert vm == "VM1_LID"

    def test_routing_summary_report_integrity(self):
        """Verify the generated 12-language routing summary report matches all invariants."""
        report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "reports", "12lang_routing_summary.json"))
        if not os.path.exists(report_path):
            pytest.skip(f"Routing summary report not found at {report_path}")

        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["total_selected_samples"] == 25134
        assert data["total_excluded_samples"] == 8603
        assert data["selected_languages_count"] == 12
        assert data["classifier_status"] == "NOT_AVAILABLE"
        assert data["lid_prediction_status"] == "NOT_AVAILABLE"
        assert data["routing_logic_status"] == "VERIFIED"
        assert data["routing_accuracy"] == 1.0
        assert data["routing_failures_count"] == 0
        assert data["final_status"] == "ROUTING LOGIC VERIFIED — LID CLASSIFIER NOT AVAILABLE"
