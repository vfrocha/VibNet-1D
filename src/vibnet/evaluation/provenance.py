"""
Reproducibility and execution provenance tracker (`src.vibnet.evaluation.provenance`).

Implements the Fase 5 specification (`docs/reproducibility.md`):
- Git `commit` SHA, `branch`, and `git_dirty` state
- Complete experiment configuration dictionary
- Command line (`argv` and string)
- Random seeds (`[42, 123, 2026]`)
- Python, platform, and ML dependency versions
- SHA-256 hashes of executed scripts and modules
- Fold accounting (`expected_folds`, `valid_folds`, `failed_folds`, `error_breakdown`)
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

TRACKED_PACKAGES: Tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "xgboost",
    "torch",
    "pytorch-tabnet",
    "PyWavelets",
    "EMD-signal",
    "matplotlib",
    "seaborn",
    "PyYAML",
)


def compute_file_sha256(path: str | Path) -> Optional[str]:
    """Return the hex SHA-256 digest of `path`, or `None` if the file does not exist."""
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git_cmd(args: Sequence[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "UNKNOWN"


def get_dependency_versions(
    packages: Iterable[str] = TRACKED_PACKAGES,
) -> Dict[str, Optional[str]]:
    """Query installed package versions via `importlib.metadata`."""
    versions: Dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def collect_run_provenance(
    project_root: str | Path,
    script_paths: Sequence[str | Path],
    seeds: Sequence[int],
    extra_config: Optional[Mapping[str, Any]] = None,
    fold_accounting: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Collect full execution provenance for an experimental run (`docs/reproducibility.md`).
    """
    root = Path(project_root).resolve()
    commit = _run_git_cmd(["rev-parse", "HEAD"], root)
    branch = _run_git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status_porcelain = _run_git_cmd(["status", "--porcelain"], root)
    git_dirty = bool(status_porcelain and status_porcelain != "UNKNOWN")

    script_hashes: Dict[str, Optional[str]] = {}
    for sp in script_paths:
        p = Path(sp)
        full_p = p if p.is_absolute() else (root / p)
        try:
            rel_key = full_p.resolve().relative_to(root).as_posix()
        except ValueError:
            rel_key = p.as_posix()
        script_hashes[rel_key] = compute_file_sha256(full_p)

    metadata: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": git_dirty,
        "command_argv": list(sys.argv),
        "command_str": " ".join(sys.argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seeds": [int(s) for s in seeds],
        "dependency_versions": get_dependency_versions(),
        "script_hashes_sha256": script_hashes,
        "config": dict(extra_config) if extra_config else {},
        "fold_accounting": dict(fold_accounting)
        if fold_accounting
        else {
            "expected_folds": 0,
            "valid_folds": 0,
            "failed_folds": 0,
            "error_breakdown": {},
        },
    }
    return metadata


def write_run_metadata(
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    filename: str = "run_metadata.json",
) -> Path:
    """Serialize `metadata` to `<output_dir>/run_metadata.json`."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(
        json.dumps(dict(metadata), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
