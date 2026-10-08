import pytest
from routing.canonical_registry import get_canonical_registry
from routing.language_group_resolver import LanguageGroupResolver


@pytest.fixture
def resolver():
    registry = get_canonical_registry()
    return LanguageGroupResolver(registry=registry)


class TestXLSRCanonicalRouting:

    def test_te_routes_to_vm2(self, resolver):
        """Telugu (te-IN) -> SOUTH -> VM2_SOUTH."""
        res = resolver.resolve("te-IN")
        assert res.group == "SOUTH"
        assert res.vm == "VM2_SOUTH"
        assert res.service_name == "vmc-south"
        assert res.port == 50052

    def test_hi_routes_to_vm3(self, resolver):
        """Hindi (hi-IN) -> NORTH -> VM3_NORTH."""
        res = resolver.resolve("hi-IN")
        assert res.group == "NORTH"
        assert res.vm == "VM3_NORTH"
        assert res.service_name == "vmc-north"
        assert res.port == 50053

    def test_bn_routes_to_vm4(self, resolver):
        """Bengali (bn-IN) -> EAST -> VM4_EAST."""
        res = resolver.resolve("bn-IN")
        assert res.group == "EAST"
        assert res.vm == "VM4_EAST"
        assert res.service_name == "vmc-east"
        assert res.port == 50054

    def test_ur_routes_to_vm5(self, resolver):
        """Urdu (ur-IN) -> NORTH_WEST_FRONTIER -> VM5_NORTH_WEST_FRONTIER."""
        res = resolver.resolve("ur-IN")
        assert res.group == "NORTH_WEST_FRONTIER"
        assert res.vm == "VM5_NORTH_WEST_FRONTIER"
        assert res.service_name == "vmc-frontier"
        assert res.port == 50055

    def test_gu_routes_to_vm6(self, resolver):
        """Gujarati (gu-IN) -> WEST -> VM6_WEST."""
        res = resolver.resolve("gu-IN")
        assert res.group == "WEST"
        assert res.vm == "VM6_WEST"
        assert res.service_name == "vmc-west"
        assert res.port == 50056

    def test_en_routes_to_vm7(self, resolver):
        """English (en-IN) -> ENGLISH -> VM7_ENGLISH."""
        res = resolver.resolve("en-IN")
        assert res.group == "ENGLISH"
        assert res.vm == "VM7_ENGLISH"
        assert res.service_name == "vmc-english"
        assert res.port == 50057

    def test_kut_routes_to_vm6(self, resolver):
        """Kutchi (kut-IN) -> WEST -> VM6_WEST."""
        res = resolver.resolve("kut-IN")
        assert res.group == "WEST"
        assert res.vm == "VM6_WEST"
        assert res.service_name == "vmc-west"
        assert res.port == 50056
