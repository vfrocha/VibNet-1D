"""
Canonical dataset registry and lightweight YAML configuration loader (`src.vibnet.data.registry`).

Centralizes metadata for the 10 rotational machinery vibration datasets used in
VibNet-1D, eliminating hardcoded sampling rates and silent fallbacks (such as the
historical 12,000 Hz fallback for `Mechanical_Gear` [5,000 Hz] and
`Electric_Motor` [50,000 Hz]).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    """Canonical specification for a vibration dataset in VibNet-1D."""

    name: str
    sampling_rate_hz: int
    role_default: str
    window_length_samples: int
    overlap_ratio: float
    normal_class: str
    reference_dataset: str
    num_conditions: int
    num_classes: int
    total_windows_audit: int
    loco_eligible: bool
    class_vocabulary: Tuple[str, ...]
    notes: str = ""


CANONICAL_DATASETS: Dict[str, DatasetSpec] = {
    "CWRU_12k": DatasetSpec(
        name="CWRU_12k",
        sampling_rate_hz=12000,
        role_default="target_or_source",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Class_0",
        reference_dataset="CWRU_12k",
        num_conditions=4,
        num_classes=4,
        total_windows_audit=2013,
        loco_eligible=True,
        class_vocabulary=("Class_0", "Class_1", "Class_2", "Class_3"),
        notes="Case Western Reserve University 12 kHz Drive End bearing dataset.",
    ),
    "CWRU_48k": DatasetSpec(
        name="CWRU_48k",
        sampling_rate_hz=48000,
        role_default="target",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Class_0",
        reference_dataset="CWRU_12k",
        num_conditions=12,
        num_classes=4,
        total_windows_audit=4524,
        loco_eligible=True,
        class_vocabulary=("Class_0", "Class_1", "Class_2", "Class_3"),
        notes=(
            "Uses CWRU_12k Class_0 as RMS reference because CWRU_48k lacks "
            "isolated healthy windows in the local repository."
        ),
    ),
    "MFPT": DatasetSpec(
        name="MFPT",
        sampling_rate_hz=48828,
        role_default="source_only",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Class_0",
        reference_dataset="MFPT",
        num_conditions=13,
        num_classes=3,
        total_windows_audit=1553,
        loco_eligible=False,
        class_vocabulary=("Class_0", "Class_1", "Class_2"),
        notes="Condition correlated with fault mode (single class per condition); source-only.",
    ),
    "PU": DatasetSpec(
        name="PU",
        sampling_rate_hz=64000,
        role_default="target",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Healthy",
        reference_dataset="PU",
        num_conditions=4,
        num_classes=4,
        total_windows_audit=10002,
        loco_eligible=True,
        class_vocabulary=("Healthy", "Inner_Race", "Outer_Race", "Compound_IR_OR"),
        notes="Paderborn University bearing dataset (4 operating conditions).",
    ),
    "UORED": DatasetSpec(
        name="UORED",
        sampling_rate_hz=42000,
        role_default="source",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Healthy",
        reference_dataset="UORED",
        num_conditions=20,
        num_classes=5,
        total_windows_audit=8200,
        loco_eligible=False,
        class_vocabulary=("Healthy", "Inner_Race", "Outer_Race", "Ball", "Cage"),
        notes="University of Ottawa Rolling-Element Dataset; 1 class per condition ID.",
    ),
    "UOEMD": DatasetSpec(
        name="UOEMD",
        sampling_rate_hz=42000,
        role_default="target",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="H",
        reference_dataset="UOEMD",
        num_conditions=16,
        num_classes=8,
        total_windows_audit=6560,
        loco_eligible=True,
        class_vocabulary=("H", "B", "I", "O", "C", "U", "M", "R"),
        notes="University of Ottawa Electric Motor Dataset (16 conditions, 8 classes).",
    ),
    "HUST": DatasetSpec(
        name="HUST",
        sampling_rate_hz=51200,
        role_default="source",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Normal",
        reference_dataset="HUST",
        num_conditions=11,
        num_classes=9,
        total_windows_audit=49500,
        loco_eligible=True,
        class_vocabulary=(
            "Normal",
            "B500",
            "B502",
            "I500",
            "I502",
            "O500",
            "O502",
            "IO500",
            "IO502",
        ),
        notes="HUST Bearing dataset; conditions N600..N800 have fewer classes than 400..500.",
    ),
    "HUST_Gearbox": DatasetSpec(
        name="HUST_Gearbox",
        sampling_rate_hz=25600,
        role_default="target",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Normal",
        reference_dataset="HUST_Gearbox",
        num_conditions=30,
        num_classes=4,
        total_windows_audit=18946,
        loco_eligible=True,
        class_vocabulary=("Normal", "Tooth_Broken", "Tooth_Miss", "Tooth_Root_Crack"),
        notes="HUST Gearbox dataset (30 balanced conditions, 4 classes).",
    ),
    "Mechanical_Gear": DatasetSpec(
        name="Mechanical_Gear",
        sampling_rate_hz=5000,
        role_default="source",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Normal",
        reference_dataset="Mechanical_Gear",
        num_conditions=4,
        num_classes=3,
        total_windows_audit=3502,
        loco_eligible=True,
        class_vocabulary=("Normal", "Gear_Fault", "Bearing_Fault"),
        notes="Canonical sampling rate is 5,000 Hz (corrects legacy 12,000 Hz fallback).",
    ),
    "Electric_Motor": DatasetSpec(
        name="Electric_Motor",
        sampling_rate_hz=50000,
        role_default="source",
        window_length_samples=2048,
        overlap_ratio=0.5,
        normal_class="Healthy",
        reference_dataset="Electric_Motor",
        num_conditions=1,
        num_classes=4,
        total_windows_audit=9600,
        loco_eligible=False,
        class_vocabulary=("Healthy", "Rotor_Fault", "Stator_Fault", "Bearing_Fault"),
        notes="Canonical sampling rate is 50,000 Hz; single condition (nominal), source-only.",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the canonical `DatasetSpec` for `name`, raising `KeyError` if unknown."""
    if name not in CANONICAL_DATASETS:
        valid = ", ".join(sorted(CANONICAL_DATASETS.keys()))
        raise KeyError(
            f"Unknown dataset '{name}'. Expected one of: {valid}. "
            "Silent sampling-rate fallbacks are disallowed."
        )
    return CANONICAL_DATASETS[name]


def get_sampling_rate_hz(name: str) -> int:
    """Return the verified sampling frequency in Hz for dataset `name`."""
    return get_dataset_spec(name).sampling_rate_hz


def list_all_datasets() -> List[str]:
    """Return all canonical dataset names in deterministic order."""
    return list(CANONICAL_DATASETS.keys())


def list_targets(cross_dataset_benchmark_only: bool = True) -> List[str]:
    """
    Return canonical target datasets.

    If `cross_dataset_benchmark_only=True`, returns the 5 benchmark targets from
    `exp_tl_cross_dataset_mlp_tabnet.py`:
    `['CWRU_12k', 'CWRU_48k', 'HUST_Gearbox', 'PU', 'UOEMD']`.
    """
    if cross_dataset_benchmark_only:
        return ["CWRU_12k", "CWRU_48k", "HUST_Gearbox", "PU", "UOEMD"]
    return [
        name
        for name, spec in CANONICAL_DATASETS.items()
        if spec.loco_eligible and "target" in spec.role_default
    ]


def list_sources(cross_dataset_benchmark_only: bool = True) -> List[str]:
    """
    Return canonical source datasets.

    If `cross_dataset_benchmark_only=True`, returns the 4 benchmark sources from
    `exp_tl_cross_dataset_mlp_tabnet.py`:
    `['UORED', 'HUST', 'Mechanical_Gear', 'Electric_Motor']`.
    """
    if cross_dataset_benchmark_only:
        return ["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]
    return [
        name
        for name, spec in CANONICAL_DATASETS.items()
        if "source" in spec.role_default
    ]


def build_reference_map() -> Dict[str, Tuple[str, str]]:
    """
    Return the mapping `dataset_name -> (reference_dataset_name, normal_class_label)`
    used for RMS reference calculation (`ADR-001`).
    """
    return {
        name: (spec.reference_dataset, spec.normal_class)
        for name, spec in CANONICAL_DATASETS.items()
    }


def _parse_scalar(val_str: str) -> Any:
    """Parse a scalar or inline list value in pure Python stdlib."""
    text = val_str.strip()
    if not text:
        return ""
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text in ("null", "None", "~"):
        return None
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            inner = text[1:-1].strip()
            if not inner:
                return []
            return [_parse_scalar(part) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _strip_yaml_comment(line: str) -> str:
    """Strip trailing `# ...` comments outside quotes."""
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            if idx == 0 or line[idx - 1].isspace():
                return line[:idx].rstrip()
    return line.rstrip()


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file (`configs/**/*.yaml`).

    Uses `PyYAML` (`yaml.safe_load`) when available, and falls back to a
    deterministic pure-Python indentation parser so configuration files can be
    inspected and unit-tested in bare Python stdlib environments.
    """
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw_text)
        return loaded if isinstance(loaded, dict) else {}
    except ImportError:
        pass

    # Pure-Python stdlib parser supporting nested dicts, lists, and scalars
    cleaned_lines: List[Tuple[int, str]] = []
    for raw_line in raw_text.splitlines():
        stripped_comment = _strip_yaml_comment(raw_line)
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        cleaned_lines.append((indent, stripped_comment.strip()))

    def parse_block(index: int, min_indent: int) -> Tuple[Any, int]:
        if index >= len(cleaned_lines):
            return {}, index
        first_indent, first_text = cleaned_lines[index]
        if first_indent < min_indent:
            return {}, index

        if first_text.startswith("- "):
            items: List[Any] = []
            while index < len(cleaned_lines):
                ind, txt = cleaned_lines[index]
                if ind < first_indent or not txt.startswith("- "):
                    break
                if ind > first_indent:
                    break
                item_body = txt[2:].strip()
                index += 1
                if not item_body:
                    sub_val, index = parse_block(index, first_indent + 1)
                    items.append(sub_val)
                elif ":" in item_body and not (
                    item_body.startswith('"') or item_body.startswith("'")
                ):
                    k, v_rest = item_body.split(":", 1)
                    item_dict: Dict[str, Any] = {}
                    v_rest = v_rest.strip()
                    if v_rest:
                        item_dict[k.strip()] = _parse_scalar(v_rest)
                    else:
                        sub_val, index = parse_block(index, first_indent + 1)
                        item_dict[k.strip()] = sub_val
                    # Check if subsequent lines belong to this dict item
                    while index < len(cleaned_lines):
                        sub_ind, sub_txt = cleaned_lines[index]
                        if sub_ind <= first_indent:
                            break
                        if ":" in sub_txt and not sub_txt.startswith("- "):
                            sk, sv = sub_txt.split(":", 1)
                            sk = sk.strip()
                            sv = sv.strip()
                            index += 1
                            if sv:
                                item_dict[sk] = _parse_scalar(sv)
                            else:
                                nested_val, index = parse_block(index, sub_ind + 1)
                                item_dict[sk] = nested_val
                        else:
                            break
                    items.append(item_dict)
                else:
                    items.append(_parse_scalar(item_body))
            return items, index

        mapping: Dict[str, Any] = {}
        while index < len(cleaned_lines):
            ind, txt = cleaned_lines[index]
            if ind < first_indent:
                break
            if ind > first_indent:
                # Unexpected extra indent; skip safely
                index += 1
                continue
            if txt.startswith("- "):
                break
            if ":" not in txt:
                index += 1
                continue
            key, val_rest = txt.split(":", 1)
            key = key.strip()
            val_rest = val_rest.strip()
            index += 1
            if val_rest:
                mapping[key] = _parse_scalar(val_rest)
            else:
                nested, index = parse_block(index, first_indent + 1)
                mapping[key] = nested
        return mapping, index

    result, _ = parse_block(0, 0)
    return result if isinstance(result, dict) else {}
