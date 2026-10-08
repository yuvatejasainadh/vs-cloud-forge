from typing import Optional, Dict
from routing.canonical_registry import get_canonical_registry, CanonicalRegistry, LanguageInfo


class LanguageGroupResolver:
    """
    Adapter that delegates directly to CanonicalRegistry.
    """

    def __init__(self, config_path: Optional[str] = None, registry: Optional[CanonicalRegistry] = None):
        self._registry: CanonicalRegistry = registry or get_canonical_registry(config_path)

    def resolve(self, language_code: str) -> Optional[LanguageInfo]:
        return self._registry.get_language_info(language_code)

    def resolve_group(self, language_code: str) -> str:
        return self._registry.resolve_group(language_code)

    def resolve_vm(self, language_code: str) -> str:
        return self._registry.resolve_vm(language_code)

    def get_language_name(self, language_code: str, fallback: str = "Unknown") -> str:
        return self._registry.get_language_name(language_code, fallback)
