import pytest
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from routing.language_group_resolver import LanguageGroupResolver
from routing.canonical_registry import get_canonical_registry


class TestLanguageGroupResolver:
    def test_canonical_group_routing(self):
        resolver = LanguageGroupResolver()

        # VM2 — South Indian (4 languages)
        assert resolver.resolve_group("te-IN") == "SOUTH"
        assert resolver.resolve_group("ta-IN") == "SOUTH"
        assert resolver.resolve_group("ml-IN") == "SOUTH"
        assert resolver.resolve_group("kn-IN") == "SOUTH"

        # VM3 — North Indian (3 languages)
        assert resolver.resolve_group("hi-IN") == "NORTH"
        assert resolver.resolve_group("mr-IN") == "NORTH"
        assert resolver.resolve_group("sa-IN") == "NORTH"

        # VM4 — East Indian (4 languages)
        assert resolver.resolve_group("bn-IN") == "EAST"
        assert resolver.resolve_group("as-IN") == "EAST"
        assert resolver.resolve_group("or-IN") == "EAST"
        assert resolver.resolve_group("bpy-IN") == "EAST"

        # VM5 — North-West / Frontier (4 languages)
        assert resolver.resolve_group("pa-IN") == "NORTH_WEST_FRONTIER"
        assert resolver.resolve_group("ks-IN") == "NORTH_WEST_FRONTIER"
        assert resolver.resolve_group("sd-IN") == "NORTH_WEST_FRONTIER"
        assert resolver.resolve_group("ur-IN") == "NORTH_WEST_FRONTIER"

        # VM6 — West Indian (3 languages)
        assert resolver.resolve_group("gu-IN") == "WEST"
        assert resolver.resolve_group("raj-IN") == "WEST"
        assert resolver.resolve_group("kut-IN") == "WEST"

        # VM7 — English Detector (1 language)
        assert resolver.resolve_group("en-IN") == "ENGLISH"

    def test_canonical_vm_routing(self):
        resolver = LanguageGroupResolver()
        assert resolver.resolve_vm("te-IN") == "VM2_SOUTH"
        assert resolver.resolve_vm("hi-IN") == "VM3_NORTH"
        assert resolver.resolve_vm("bn-IN") == "VM4_EAST"
        assert resolver.resolve_vm("pa-IN") == "VM5_NORTH_WEST_FRONTIER"
        assert resolver.resolve_vm("gu-IN") == "VM6_WEST"
        assert resolver.resolve_vm("en-IN") == "VM7_ENGLISH"

    def test_unknown_language_fallback(self):
        resolver = LanguageGroupResolver()
        assert resolver.resolve_group("unknown-LANG") == "UNKNOWN_GROUP"
        assert resolver.resolve_group("") == "UNKNOWN_GROUP"
        assert resolver.resolve_group(None) == "UNKNOWN_GROUP"
