import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import yaml
from telemetry.logging import get_logger

logger = get_logger("canonical_registry")


@dataclass(frozen=True)
class LanguageInfo:
    code: str
    name: str
    native_name: str
    family: str
    group: str
    vm: str
    service_name: str
    port: int
    is_canonical: bool = True


@dataclass(frozen=True)
class GroupInfo:
    group_id: str
    vm_id: str
    name: str
    description: str
    languages: List[str]


@dataclass(frozen=True)
class VMInfo:
    vm_id: str
    role: str
    service_name: str
    port: int
    status: str
    model_type: Optional[str] = None


class CanonicalRegistry:
    """
    Authoritative VoiceShield V2 Canonical Language & Routing Registry.
    Guarantees strict bidirectional mapping for 19 canonical languages, 6 detector groups, and 7 VMs.
    """

    EXPECTED_CANONICAL_COUNT = 19
    EXPECTED_GROUP_COUNT = 6
    EXPECTED_VM_COUNT = 7

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_default_config()
        self.version: str = "2.0.0"
        self.default_group: str = "UNKNOWN_GROUP"
        self.default_vm: str = "VM1_LID"

        self._languages: Dict[str, LanguageInfo] = {}
        self._groups: Dict[str, GroupInfo] = {}
        self._vms: Dict[str, VMInfo] = {}

        self.load_and_validate()

    @staticmethod
    def _find_default_config() -> str:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "configs", "canonical_registry.yaml"),
            os.path.join(os.getcwd(), "services", "vmc-lid", "configs", "canonical_registry.yaml"),
            os.path.join(os.getcwd(), "configs", "canonical_registry.yaml"),
            "/app/configs/canonical_registry.yaml",
        ]
        for path in candidates:
            normalized = os.path.abspath(path)
            if os.path.exists(normalized):
                return normalized
        return candidates[0]

    def load_and_validate(self) -> None:
        """Loads canonical registry from YAML and validates integrity."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Canonical registry YAML not found at '{self.config_path}'. Applying hardcoded canonical baseline.")
            self._apply_canonical_defaults()
            self._validate_integrity()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.version = data.get("version", "2.0.0")
            self.default_group = data.get("default_group", "UNKNOWN_GROUP")
            self.default_vm = data.get("default_vm", "VM1_LID")

            vms_data = data.get("vms", {})
            self._vms.clear()
            for vm_id, v in vms_data.items():
                self._vms[vm_id] = VMInfo(
                    vm_id=vm_id,
                    role=v.get("role", "DETECTOR"),
                    service_name=v.get("service_name", ""),
                    port=int(v.get("port", 0)),
                    status=v.get("status", "PLANNED"),
                    model_type=v.get("model_type"),
                )

            groups_data = data.get("groups", {})
            self._groups.clear()
            for g_id, g in groups_data.items():
                self._groups[g_id] = GroupInfo(
                    group_id=g_id,
                    vm_id=g.get("vm", ""),
                    name=g.get("name", g_id),
                    description=g.get("description", ""),
                    languages=list(g.get("languages", [])),
                )

            lang_data = data.get("languages", {})
            self._languages.clear()
            for code, info in lang_data.items():
                vm_id = info.get("vm", "")
                vm_meta = self._vms.get(vm_id)
                service_name = vm_meta.service_name if vm_meta else ""
                port = vm_meta.port if vm_meta else 0

                self._languages[code] = LanguageInfo(
                    code=code,
                    name=info.get("name", code),
                    native_name=info.get("native_name", ""),
                    family=info.get("family", ""),
                    group=info.get("group", ""),
                    vm=vm_id,
                    service_name=service_name,
                    port=port,
                    is_canonical=True,
                )

            self._validate_integrity()
            logger.info(
                f"Canonical registry v{self.version} loaded successfully: "
                f"{len(self._languages)} languages, {len(self._groups)} groups, {len(self._vms)} VMs."
            )
        except Exception as e:
            logger.error(f"Error loading canonical registry config: {e}. Falling back to default.")
            self._apply_canonical_defaults()
            self._validate_integrity()

    def _apply_canonical_defaults(self) -> None:
        """In-memory authoritative baseline if YAML is unavailable."""
        self._vms = {
            "VM1_LID": VMInfo("VM1_LID", "LANGUAGE_IDENTIFICATION", "vmc-lid", 50051, "ACTIVE"),
            "VM2_SOUTH": VMInfo("VM2_SOUTH", "INDIC_DETECTOR", "vmc-south", 50052, "PLANNED", "MULTILINGUAL_AASIST_L"),
            "VM3_NORTH": VMInfo("VM3_NORTH", "INDIC_DETECTOR", "vmc-north", 50053, "PLANNED", "MULTILINGUAL_AASIST_L"),
            "VM4_EAST": VMInfo("VM4_EAST", "INDIC_DETECTOR", "vmc-east", 50054, "PLANNED", "MULTILINGUAL_AASIST_L"),
            "VM5_NORTH_WEST_FRONTIER": VMInfo("VM5_NORTH_WEST_FRONTIER", "INDIC_DETECTOR", "vmc-frontier", 50055, "PLANNED", "MULTILINGUAL_AASIST_L"),
            "VM6_WEST": VMInfo("VM6_WEST", "INDIC_DETECTOR", "vmc-west", 50056, "PLANNED", "MULTILINGUAL_AASIST_L"),
            "VM7_ENGLISH": VMInfo("VM7_ENGLISH", "ENGLISH_DETECTOR", "vmc-english", 50057, "PLANNED", "AZURE_AASIST_L"),
        }

        self._groups = {
            "SOUTH": GroupInfo("SOUTH", "VM2_SOUTH", "South Indian Group", "Dravidian languages", ["te-IN", "ta-IN", "ml-IN", "kn-IN"]),
            "NORTH": GroupInfo("NORTH", "VM3_NORTH", "North Indian Group", "Central & Northern Indo-Aryan", ["hi-IN", "mr-IN", "sa-IN"]),
            "EAST": GroupInfo("EAST", "VM4_EAST", "East Indian Group", "Eastern Indo-Aryan", ["bn-IN", "as-IN", "or-IN", "bpy-IN"]),
            "NORTH_WEST_FRONTIER": GroupInfo("NORTH_WEST_FRONTIER", "VM5_NORTH_WEST_FRONTIER", "North-West / Frontier Group", "North-Western Indo-Aryan & Dardic", ["pa-IN", "ks-IN", "sd-IN", "ur-IN"]),
            "WEST": GroupInfo("WEST", "VM6_WEST", "West Indian Group", "Western Indo-Aryan", ["gu-IN", "raj-IN", "kut-IN"]),
            "ENGLISH": GroupInfo("ENGLISH", "VM7_ENGLISH", "English", "Azure AASIST-L detector", ["en-IN"]),
        }

        lang_defs = [
            ("te-IN", "Telugu", "తెలుగు", "Dravidian", "SOUTH", "VM2_SOUTH"),
            ("ta-IN", "Tamil", "தமிழ்", "Dravidian", "SOUTH", "VM2_SOUTH"),
            ("ml-IN", "Malayalam", "മലയാളം", "Dravidian", "SOUTH", "VM2_SOUTH"),
            ("kn-IN", "Kannada", "ಕನ್ನಡ", "Dravidian", "SOUTH", "VM2_SOUTH"),
            ("hi-IN", "Hindi", "हिन्दी", "Indo-Aryan", "NORTH", "VM3_NORTH"),
            ("mr-IN", "Marathi", "मराठी", "Indo-Aryan", "NORTH", "VM3_NORTH"),
            ("sa-IN", "Sanskrit", "संस्कृतम्", "Indo-Aryan", "NORTH", "VM3_NORTH"),
            ("bn-IN", "Bengali", "বাংলা", "Indo-Aryan", "EAST", "VM4_EAST"),
            ("as-IN", "Assamese", "অসমীয়া", "Indo-Aryan", "EAST", "VM4_EAST"),
            ("or-IN", "Odia", "ଓଡ଼ିଆ", "Indo-Aryan", "EAST", "VM4_EAST"),
            ("bpy-IN", "Bishnupriya Manipuri", "বিষ্ণুপ্রিয়া মণিপুরী", "Indo-Aryan", "EAST", "VM4_EAST"),
            ("pa-IN", "Punjabi", "ਪੰਜਾਬੀ", "Indo-Aryan", "NORTH_WEST_FRONTIER", "VM5_NORTH_WEST_FRONTIER"),
            ("ks-IN", "Kashmiri", "कॉशुर", "Dardic / Indo-Aryan", "NORTH_WEST_FRONTIER", "VM5_NORTH_WEST_FRONTIER"),
            ("sd-IN", "Sindhi", "سنڌي / सिन्धी", "Indo-Aryan", "NORTH_WEST_FRONTIER", "VM5_NORTH_WEST_FRONTIER"),
            ("ur-IN", "Urdu", "اردو", "Indo-Aryan", "NORTH_WEST_FRONTIER", "VM5_NORTH_WEST_FRONTIER"),
            ("gu-IN", "Gujarati", "ગુજરાતી", "Indo-Aryan", "WEST", "VM6_WEST"),
            ("raj-IN", "Rajasthani", "राजस्थानी", "Indo-Aryan", "WEST", "VM6_WEST"),
            ("kut-IN", "Kutchi", "કચ્છી", "Indo-Aryan", "WEST", "VM6_WEST"),
            ("en-IN", "English", "English", "Germanic", "ENGLISH", "VM7_ENGLISH"),
        ]

        self._languages.clear()
        for code, name, native_name, family, group, vm in lang_defs:
            vm_meta = self._vms[vm]
            self._languages[code] = LanguageInfo(
                code=code,
                name=name,
                native_name=native_name,
                family=family,
                group=group,
                vm=vm,
                service_name=vm_meta.service_name,
                port=vm_meta.port,
                is_canonical=True,
            )

    def _validate_integrity(self) -> None:
        """Validates that all 19 languages map cleanly and bidirectionally."""
        if len(self._languages) != self.EXPECTED_CANONICAL_COUNT:
            raise ValueError(
                f"Registry validation failure: expected {self.EXPECTED_CANONICAL_COUNT} canonical languages, got {len(self._languages)}"
            )

        if len(self._groups) != self.EXPECTED_GROUP_COUNT:
            raise ValueError(
                f"Registry validation failure: expected {self.EXPECTED_GROUP_COUNT} groups, got {len(self._groups)}"
            )

        for lang_code, lang_info in self._languages.items():
            if lang_info.group not in self._groups:
                raise ValueError(f"Language {lang_code} has unmapped group: {lang_info.group}")
            if lang_info.vm not in self._vms:
                raise ValueError(f"Language {lang_code} has unmapped VM: {lang_info.vm}")
            group_info = self._groups[lang_info.group]
            if lang_code not in group_info.languages:
                raise ValueError(f"Group {lang_info.group} does not list language {lang_code}")

    def get_language_info(self, language_code: str) -> Optional[LanguageInfo]:
        """Returns metadata for an ISO language code."""
        return self._languages.get(language_code)

    def resolve_group(self, language_code: str) -> str:
        """Returns semantic group name (e.g., 'SOUTH', 'NORTH') for a language code."""
        info = self._languages.get(language_code)
        return info.group if info else self.default_group

    def resolve_vm(self, language_code: str) -> str:
        """Returns target VM ID (e.g., 'VM2_SOUTH') for a language code."""
        info = self._languages.get(language_code)
        return info.vm if info else self.default_vm

    def get_language_name(self, language_code: str, fallback: str = "Unknown") -> str:
        """Returns human-readable name for an ISO language code."""
        info = self._languages.get(language_code)
        return info.name if info else fallback

    def get_group_info(self, group_id: str) -> Optional[GroupInfo]:
        """Returns metadata for a group."""
        return self._groups.get(group_id)

    def get_vm_info(self, vm_id: str) -> Optional[VMInfo]:
        """Returns metadata for a VM."""
        return self._vms.get(vm_id)

    def list_canonical_languages(self) -> List[LanguageInfo]:
        """Returns all 19 canonical languages."""
        return list(self._languages.values())

    def list_groups(self) -> List[GroupInfo]:
        """Returns all 6 detector groups."""
        return list(self._groups.values())

    def list_vms(self) -> List[VMInfo]:
        """Returns all 7 VMs."""
        return list(self._vms.values())

    def lid_labels(self) -> List[str]:
        """Returns ordered list of the 19 canonical ISO language codes."""
        return [lang.code for lang in self.list_canonical_languages()]


# Global singleton instance
_canonical_registry_instance: Optional[CanonicalRegistry] = None


def get_canonical_registry(config_path: Optional[str] = None) -> CanonicalRegistry:
    global _canonical_registry_instance
    if _canonical_registry_instance is None or config_path is not None:
        _canonical_registry_instance = CanonicalRegistry(config_path)
    return _canonical_registry_instance
