import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from routing.canonical_registry import CanonicalRegistry, get_canonical_registry


class TestCanonicalRegistry:
    def test_canonical_19_languages_presence(self):
        reg = get_canonical_registry()
        langs = reg.list_canonical_languages()
        assert len(langs) == 19

        codes = {l.code for l in langs}
        expected_19 = {
            "te-IN", "ta-IN", "ml-IN", "kn-IN",
            "hi-IN", "mr-IN", "sa-IN",
            "bn-IN", "as-IN", "or-IN", "bpy-IN",
            "pa-IN", "ks-IN", "sd-IN", "ur-IN",
            "gu-IN", "raj-IN", "kut-IN",
            "en-IN"
        }
        assert codes == expected_19

    def test_canonical_6_detector_groups(self):
        reg = get_canonical_registry()
        groups = reg.list_groups()
        assert len(groups) == 6

        group_ids = {g.group_id for g in groups}
        assert group_ids == {"SOUTH", "NORTH", "EAST", "NORTH_WEST_FRONTIER", "WEST", "ENGLISH"}

    def test_canonical_7_vms(self):
        reg = get_canonical_registry()
        vms = reg.list_vms()
        assert len(vms) == 7

        vm_ids = {v.vm_id for v in vms}
        expected_vms = {
            "VM1_LID", "VM2_SOUTH", "VM3_NORTH",
            "VM4_EAST", "VM5_NORTH_WEST_FRONTIER", "VM6_WEST", "VM7_ENGLISH"
        }
        assert vm_ids == expected_vms

    def test_bidirectional_integrity(self):
        reg = get_canonical_registry()
        for l in reg.list_canonical_languages():
            assert l.group in {"SOUTH", "NORTH", "EAST", "NORTH_WEST_FRONTIER", "WEST", "ENGLISH"}
            assert l.vm in {"VM2_SOUTH", "VM3_NORTH", "VM4_EAST", "VM5_NORTH_WEST_FRONTIER", "VM6_WEST", "VM7_ENGLISH"}
            
            group_info = reg.get_group_info(l.group)
            assert group_info is not None
            assert l.code in group_info.languages

            vm_info = reg.get_vm_info(l.vm)
            assert vm_info is not None
            assert vm_info.port > 0
