import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from routing.language_group_resolver import LanguageGroupResolver
from runtimes.open_vakgyata_runtime import OpenVakgyataRuntime


class TestLanguageMapping:
    EXPECTED_LANGUAGES = {
        "en-IN": "English",
        "hi-IN": "Hindi",
        "or-IN": "Odia",
        "bn-IN": "Bengali",
        "ta-IN": "Tamil",
        "te-IN": "Telugu",
        "kn-IN": "Kannada",
        "ml-IN": "Malayalam",
        "mr-IN": "Marathi",
        "gu-IN": "Gujarati"
    }

    def test_all_10_indic_languages_defined(self):
        resolver = LanguageGroupResolver()
        for code, expected_name in self.EXPECTED_LANGUAGES.items():
            name = resolver.get_language_name(code)
            assert name == expected_name, f"Expected {expected_name} for code {code}, got {name}"

    def test_runtime_default_id2label_mapping(self):
        id2label = OpenVakgyataRuntime.DEFAULT_ID2LABEL
        assert len(id2label) == 10
        for idx in range(10):
            assert idx in id2label
            assert id2label[idx] in self.EXPECTED_LANGUAGES

    def test_unsupported_language_fallback(self):
        resolver = LanguageGroupResolver()
        assert resolver.get_language_name("fr-FR", fallback="Unknown") == "Unknown"
        assert resolver.get_language_name("xx-YY", fallback="CustomFallback") == "CustomFallback"
