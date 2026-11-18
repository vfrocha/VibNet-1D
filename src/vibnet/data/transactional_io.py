"""
Transactional dataset and manifest I/O primitives (`src.vibnet.data.transactional_io`).

Provides safe, atomic staging, pre-replacement structural/numerical validation,
same-filesystem and anti-symlink guards, backup creation, and automatic rollback
for both dataset window directories (`.npy`) and manifest files (`.csv`, `.json`).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Union

import numpy as np

from src.vibnet.data.registry import CANONICAL_DATASETS


def _utc_timestamp_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _compute_file_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def assert_no_symlink_components(path: Path, stop_at: Optional[Path] = None) -> None:
    """
    Verify that neither `path` nor any of its parent components is a symbolic link.

    Raises `PermissionError` if any component is a symlink.
    """
    raw_path = Path(path).expanduser()
    stop_resolved = Path(stop_at).expanduser().resolve() if stop_at is not None else None

    current = raw_path
    checked: Set[str] = set()
    while True:
        key = str(current)
        if key in checked:
            break
        checked.add(key)

        if current.is_symlink():
            raise PermissionError(
                f"Link simbólico detectado e bloqueado por segurança: '{current}'"
            )

        if stop_resolved is not None and current.exists():
            try:
                if current.resolve() == stop_resolved:
                    break
            except OSError:
                pass

        if current.parent == current:
            break
        current = current.parent


def assert_no_symlinks_in_tree(root_dir: Path) -> None:
    """
    Verify that `root_dir` has no symlink components and contains no symlinked
    files or subdirectories at any depth.
    """
    assert_no_symlink_components(root_dir)
    if not root_dir.exists():
        return
    if root_dir.is_symlink():
        raise PermissionError(
            f"O diretório raiz '{root_dir}' é um link simbólico (não permitido)."
        )

    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
        base = Path(dirpath)
        for dname in dirnames:
            candidate = base / dname
            if candidate.is_symlink():
                raise PermissionError(
                    f"Subdiretório simbólico detectado dentro da árvore de dados: '{candidate}'"
                )
        for fname in filenames:
            candidate = base / fname
            if candidate.is_symlink():
                raise PermissionError(
                    f"Arquivo simbólico detectado dentro da árvore de dados: '{candidate}'"
                )


def get_device_id(path: Path) -> int:
    """
    Return the filesystem device ID (`st_dev`) of the nearest existing ancestor of `path`
    after verifying that no component is a symbolic link.
    """
    assert_no_symlink_components(path)
    probe = Path(path).expanduser().resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return int(os.stat(probe, follow_symlinks=False).st_dev)


def assert_same_filesystem(*paths: Path) -> int:
    """
    Ensure all given paths (staging, target, backup, manifest) reside on the
    exact same filesystem (`st_dev`) so `Path.replace()` is guaranteed atomic.
    """
    if not paths:
        raise ValueError("Pelo menos um caminho deve ser informado para verificação de filesystem.")

    dev_by_path: Dict[str, int] = {}
    for p in paths:
        if p is not None:
            dev_by_path[str(p)] = get_device_id(Path(p))

    unique_devs = set(dev_by_path.values())
    if len(unique_devs) > 1:
        details = ", ".join(f"{k} (st_dev={v})" for k, v in dev_by_path.items())
        raise RuntimeError(
            f"Operação transacional abortada: staging, backup e destino estão em filesystems "
            f"distintos ({details}). Todos devem residir no mesmo filesystem para substituição atômica."
        )
    return next(iter(unique_devs))


def assert_safe_dataset_target_path(
    target_dir: Path,
    allowed_variant_root: Path,
    dataset_name: str,
    allowed_datasets: Optional[Sequence[str]] = None,
) -> Path:
    """
    Validate that `target_dir` is strictly `<allowed_variant_root>/<dataset_name>`,
    free of symlinks, path traversal tokens, or unauthorized dataset names.
    """
    valid_names = set(allowed_datasets) if allowed_datasets is not None else set(CANONICAL_DATASETS.keys())
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("Nome de dataset vazio ou inválido.")
    if (
        "/" in dataset_name
        or "\\" in dataset_name
        or ".." in dataset_name
        or dataset_name.startswith(".")
    ):
        raise ValueError(f"Nome de dataset contém caracteres de navegação proibidos: '{dataset_name}'")
    if dataset_name not in valid_names:
        raise ValueError(
            f"Dataset '{dataset_name}' fora da lista autorizada: {sorted(valid_names)}"
        )

    assert_no_symlink_components(allowed_variant_root)
    assert_no_symlink_components(target_dir)

    resolved_root = Path(allowed_variant_root).expanduser().resolve()
    resolved_target = Path(target_dir).expanduser().resolve()

    if not resolved_target.is_relative_to(resolved_root):
        raise PermissionError(
            f"Violação de contenção de caminho: '{resolved_target}' está fora de '{resolved_root}'."
        )
    if resolved_target == resolved_root or resolved_target.parent != resolved_root:
        raise PermissionError(
            f"Caminho alvo inválido: '{resolved_target}' deve ser filho direto de '{resolved_root}'."
        )
    if resolved_target.name != dataset_name:
        raise PermissionError(
            f"Inconsistência de nome de diretório alvo: '{resolved_target.name}' != '{dataset_name}'."
        )

    return resolved_target


def create_staging_dataset_dir(
    allowed_variant_root: Path,
    dataset_name: str,
    backup_root: Optional[Path] = None,
    allowed_datasets: Optional[Sequence[str]] = None,
) -> Path:
    """
    Create an isolated temporary staging directory inside `allowed_variant_root`
    on the same filesystem as the target dataset directory and `backup_root`.
    """
    target_dir = Path(allowed_variant_root) / dataset_name
    assert_safe_dataset_target_path(
        target_dir=target_dir,
        allowed_variant_root=allowed_variant_root,
        dataset_name=dataset_name,
        allowed_datasets=allowed_datasets,
    )
    resolved_root = Path(allowed_variant_root).expanduser().resolve()
    resolved_root.mkdir(parents=True, exist_ok=True)

    paths_to_check = [resolved_root, target_dir]
    if backup_root is not None:
        assert_no_symlink_components(backup_root)
        paths_to_check.append(Path(backup_root))
    assert_same_filesystem(*paths_to_check)

    ts = _utc_timestamp_token()
    staging_dir = resolved_root / f".staging_{dataset_name}_{ts}_{os.getpid()}"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=False)
    return staging_dir


def cleanup_staging_dir(staging_dir: Optional[Path]) -> None:
    """Safely remove a temporary `.staging_*` directory if it still exists."""
    if staging_dir is None:
        return
    assert_no_symlink_components(staging_dir)
    resolved = Path(staging_dir).expanduser().resolve()
    if not resolved.name.startswith(".staging_"):
        raise PermissionError(
            f"Recusando remover diretório que não é de staging: '{resolved}'"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def validate_staged_dataset_dir(
    staging_dir: Path,
    expected_window_count: Optional[int] = None,
    expected_window_len: Optional[int] = None,
    reference_rms: Optional[float] = None,
    source_dir: Optional[Path] = None,
    expected_relative_paths: Optional[Sequence[str | Path]] = None,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    Perform complete structural, path-equivalence, and numerical validation on
    a staging dataset directory before allowing it to replace the active directory.

    Checks:
      1. No symlinks anywhere in `staging_dir` (or `source_dir` if provided).
      2. Only `.npy` files exist, with non-zero count matching `expected_window_count`.
      3. Every `.npy` file has strict relative depth 3 (`<condition>/<class>/<window.npy>`).
      4. Exact 1-to-1 relative path set equality against `source_dir` and/or
         `expected_relative_paths` (Adjustment 1).
      5. Every `.npy` array is readable (`allow_pickle=False`), finite (`NaN`/`Inf`-free),
         has `RMS > 0`, matches `expected_window_len` (when given), and satisfies
         `RMS(staging) == RMS(source) / reference_rms` within tolerance (when given).
    """
    try:
        assert_no_symlinks_in_tree(staging_dir)
        if source_dir is not None:
            assert_no_symlinks_in_tree(source_dir)
    except Exception as exc:
        return {
            "valid": False,
            "window_count": 0,
            "reason": f"Falha de segurança contra links simbólicos: {exc}",
        }

    if not staging_dir.is_dir():
        return {
            "valid": False,
            "window_count": 0,
            "reason": f"Diretório de staging ausente: {staging_dir}",
        }

    all_files = sorted(p for p in staging_dir.rglob("*") if p.is_file())
    non_npy = [str(p.relative_to(staging_dir)) for p in all_files if p.suffix != ".npy"]
    if non_npy:
        return {
            "valid": False,
            "window_count": 0,
            "reason": f"Arquivos não-.npy inesperados no staging: {non_npy[:5]}",
        }

    npy_files = sorted(staging_dir.rglob("*.npy"))
    if not npy_files:
        return {
            "valid": False,
            "window_count": 0,
            "reason": f"Nenhum arquivo .npy encontrado no staging: {staging_dir}",
        }

    if expected_window_count is not None and len(npy_files) != expected_window_count:
        return {
            "valid": False,
            "window_count": len(npy_files),
            "reason": (
                f"Contagem de janelas divergente no staging: "
                f"obtido {len(npy_files)} != esperado {expected_window_count}."
            ),
        }

    anomalous_depth = [
        p.relative_to(staging_dir).as_posix()
        for p in npy_files
        if len(p.relative_to(staging_dir).parts) != 3
    ]
    if anomalous_depth:
        return {
            "valid": False,
            "window_count": len(npy_files),
            "reason": (
                f"Estrutura hierárquica inválida no staging ({len(anomalous_depth)} arquivos "
                f"fora do padrão <condição>/<classe>/<janela.npy>): {anomalous_depth[:5]}"
            ),
        }

    staging_rel_map: Dict[str, Path] = {
        p.relative_to(staging_dir).as_posix(): p for p in npy_files
    }
    staging_rel_set: Set[str] = set(staging_rel_map.keys())

    # Ajuste Técnico 1: Comparação exata do conjunto de caminhos relativos com source_dir
    source_rel_map: Optional[Dict[str, Path]] = None
    if source_dir is not None:
        if not source_dir.is_dir():
            return {
                "valid": False,
                "window_count": len(npy_files),
                "reason": f"Diretório de origem ausente para comparação: {source_dir}",
            }
        source_files = sorted(source_dir.rglob("*.npy"))
        source_rel_map = {
            p.relative_to(source_dir).as_posix(): p for p in source_files
        }
        source_rel_set = set(source_rel_map.keys())
        if staging_rel_set != source_rel_set:
            missing_in_staging = sorted(source_rel_set - staging_rel_set)
            extra_in_staging = sorted(staging_rel_set - source_rel_set)
            return {
                "valid": False,
                "window_count": len(npy_files),
                "missing_relative_paths": missing_in_staging[:10],
                "extra_relative_paths": extra_in_staging[:10],
                "reason": (
                    f"Caminhos relativos divergentes entre entrada e saída: "
                    f"faltantes={len(missing_in_staging)} ({missing_in_staging[:3]}), "
                    f"extras={len(extra_in_staging)} ({extra_in_staging[:3]})."
                ),
            }

    if expected_relative_paths is not None:
        expected_rel_set = {Path(p).as_posix() for p in expected_relative_paths}
        if staging_rel_set != expected_rel_set:
            missing_rel = sorted(expected_rel_set - staging_rel_set)
            extra_rel = sorted(staging_rel_set - expected_rel_set)
            return {
                "valid": False,
                "window_count": len(npy_files),
                "missing_relative_paths": missing_rel[:10],
                "extra_relative_paths": extra_rel[:10],
                "reason": (
                    f"Caminhos relativos divergentes do esperado: "
                    f"faltantes={len(missing_rel)} ({missing_rel[:3]}), "
                    f"extras={len(extra_rel)} ({extra_rel[:3]})."
                ),
            }

    conditions = sorted({p.parent.parent.name for p in npy_files})
    classes = sorted({p.parent.name for p in npy_files})

    for rel_posix, fp in staging_rel_map.items():
        try:
            arr = np.load(fp, allow_pickle=False)
        except Exception as exc:
            return {
                "valid": False,
                "window_count": len(npy_files),
                "reason": f"Arquivo .npy corrompido no staging ({rel_posix}): {exc}",
            }

        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            return {
                "valid": False,
                "window_count": len(npy_files),
                "reason": f"Sinal vazio (0 amostras) em {rel_posix}.",
            }
        if expected_window_len is not None and flat.size != expected_window_len:
            return {
                "valid": False,
                "window_count": len(npy_files),
                "reason": (
                    f"Comprimento de janela divergente em {rel_posix}: "
                    f"obtido {flat.size} != esperado {expected_window_len}."
                ),
            }
        if not np.all(np.isfinite(flat)):
            return {
                "valid": False,
                "window_count": len(npy_files),
                "reason": f"Valores NaN/Inf detectados em {rel_posix}.",
            }

        rms_val = float(np.sqrt(np.mean(np.square(flat))))
        if not np.isfinite(rms_val) or rms_val <= 0.0:
            return {
                "valid": False,
                "window_count": len(npy_files),
                "reason": f"RMS inválido ou zero ({rms_val}) em {rel_posix}.",
            }

        if source_rel_map is not None and reference_rms is not None:
            src_fp = source_rel_map[rel_posix]
            src_arr = np.load(src_fp, allow_pickle=False)
            if arr.shape != src_arr.shape:
                return {
                    "valid": False,
                    "window_count": len(npy_files),
                    "reason": (
                        f"Shape divergente entre entrada e saída em {rel_posix}: "
                        f"{src_arr.shape} -> {arr.shape}."
                    ),
                }
            if arr.dtype != src_arr.dtype:
                return {
                    "valid": False,
                    "window_count": len(npy_files),
                    "reason": (
                        f"Dtype divergente entre entrada e saída em {rel_posix}: "
                        f"{src_arr.dtype} -> {arr.dtype}."
                    ),
                }
            src_flat = np.asarray(src_arr, dtype=np.float64).reshape(-1)
            src_rms = float(np.sqrt(np.mean(np.square(src_flat))))
            expected_rms = src_rms / float(reference_rms)
            abs_err = abs(rms_val - expected_rms)
            if abs_err > (abs_tol + rel_tol * abs(expected_rms)):
                return {
                    "valid": False,
                    "window_count": len(npy_files),
                    "reason": (
                        f"Falha matemática na normalização RMS em {rel_posix}: "
                        f"obtido {rms_val:.12g} != esperado {expected_rms:.12g} (erro={abs_err:.3e})."
                    ),
                }

    return {
        "valid": True,
        "window_count": len(npy_files),
        "conditions": conditions,
        "classes": classes,
        "relative_paths_count": len(staging_rel_set),
        "reason": f"{len(npy_files)} janelas verificadas integralmente no staging.",
    }


def _prune_empty_backup_parents(start_dir: Optional[Path], stop_at: Optional[Path]) -> None:
    """Remove empty timestamp/variant directories inside `backup_root` up to `stop_at`."""
    if start_dir is None:
        return
    stop_resolved = Path(stop_at).expanduser().resolve() if stop_at is not None else None
    current = Path(start_dir).expanduser().resolve()
    while current.exists() and current.is_dir():
        if stop_resolved is not None and current == stop_resolved:
            break
        if stop_resolved is not None and not current.is_relative_to(stop_resolved):
            break
        try:
            next(current.iterdir())
            break
        except StopIteration:
            parent = current.parent
            current.rmdir()
            if parent == current:
                break
            current = parent


@dataclass
class DatasetCommitReceipt:
    """Receipt for a committed dataset directory swap, supporting atomic rollback and deferred backup cleanup."""

    dataset_name: str
    variant_name: str
    target_dir: Path
    backup_path: Optional[Path]
    had_previous_target: bool
    keep_backup: bool = True
    backup_root: Optional[Path] = None
    committed: bool = True
    finalized: bool = False

    def rollback(self) -> None:
        """Restore `target_dir` to its exact pre-commit state using `backup_path`."""
        if not self.committed:
            return
        if self.target_dir.exists():
            assert_no_symlinks_in_tree(self.target_dir)
            shutil.rmtree(self.target_dir)
        if self.had_previous_target:
            if self.backup_path is None or not self.backup_path.exists():
                raise RuntimeError(
                    f"Rollback impossível para '{self.target_dir}': backup '{self.backup_path}' não encontrado."
                )
            assert_no_symlinks_in_tree(self.backup_path)
            backup_parent = self.backup_path.parent
            self.backup_path.replace(self.target_dir)
            _prune_empty_backup_parents(backup_parent, self.backup_root)
        self.committed = False

    def finalize(self, keep_backup: Optional[bool] = None) -> None:
        """
        Finalize the commit after the entire logical transaction succeeds.
        Only discards `backup_path` here when `keep_backup` is False.
        """
        if not self.committed or self.finalized:
            return
        effective_keep = self.keep_backup if keep_backup is None else bool(keep_backup)
        if not effective_keep and self.backup_path is not None:
            backup_parent = self.backup_path.parent
            if self.backup_path.exists():
                assert_no_symlinks_in_tree(self.backup_path)
                shutil.rmtree(self.backup_path)
            _prune_empty_backup_parents(backup_parent, self.backup_root)
            self.backup_path = None
        self.finalized = True


def commit_staged_dataset_dir(
    staging_dir: Path,
    target_dir: Path,
    allowed_variant_root: Path,
    backup_root: Path,
    dataset_name: str,
    variant_name: str,
    allowed_datasets: Optional[Sequence[str]] = None,
    keep_backup: bool = True,
) -> DatasetCommitReceipt:
    """
    Atomically replace `target_dir` with `staging_dir` after moving any existing
    `target_dir` to `backup_root`. Automatically rolls back if the swap fails.

    Note: The backup at `backup_path` is always preserved while the transaction
    is in flight (even if `keep_backup=False`) and is only discarded when
    `receipt.finalize()` / `finalize_transaction_receipts()` is called after
    all directories and manifests have been committed.
    """
    resolved_target = assert_safe_dataset_target_path(
        target_dir=target_dir,
        allowed_variant_root=allowed_variant_root,
        dataset_name=dataset_name,
        allowed_datasets=allowed_datasets,
    )
    assert_no_symlinks_in_tree(staging_dir)
    assert_no_symlink_components(backup_root)

    resolved_staging = Path(staging_dir).expanduser().resolve()
    resolved_backup_root = Path(backup_root).expanduser().resolve()
    assert_same_filesystem(resolved_staging, resolved_target, resolved_backup_root)

    had_previous = resolved_target.exists()
    backup_path: Optional[Path] = None

    if had_previous:
        assert_no_symlinks_in_tree(resolved_target)
        ts = _utc_timestamp_token()
        backup_parent = resolved_backup_root / ts / variant_name
        backup_parent.mkdir(parents=True, exist_ok=True)
        backup_path = backup_parent / dataset_name
        resolved_target.replace(backup_path)

    try:
        resolved_staging.replace(resolved_target)
    except Exception as exc:
        if had_previous and backup_path is not None and backup_path.exists() and not resolved_target.exists():
            backup_path.replace(resolved_target)
        raise RuntimeError(
            f"Falha ao promover staging para '{resolved_target}'; versão anterior restaurada. Erro: {exc}"
        ) from exc

    return DatasetCommitReceipt(
        dataset_name=dataset_name,
        variant_name=variant_name,
        target_dir=resolved_target,
        backup_path=backup_path,
        had_previous_target=had_previous,
        keep_backup=keep_backup,
        backup_root=resolved_backup_root,
        committed=True,
        finalized=False,
    )


@dataclass
class ManifestCommitReceipt:
    """Receipt for a transactional manifest update, supporting atomic rollback and deferred backup cleanup."""

    manifest_path: Path
    backup_path: Optional[Path]
    had_previous_manifest: bool
    keep_backup: bool = True
    backup_root: Optional[Path] = None
    committed: bool = True
    finalized: bool = False

    def rollback(self) -> None:
        """Restore `manifest_path` from `backup_path` (or remove if newly created)."""
        if not self.committed:
            return
        if self.had_previous_manifest:
            if self.backup_path is None or not self.backup_path.is_file():
                raise RuntimeError(
                    f"Rollback impossível para manifesto '{self.manifest_path}': "
                    f"backup '{self.backup_path}' não encontrado."
                )
            assert_no_symlink_components(self.backup_path)
            shutil.copy2(self.backup_path, self.manifest_path)
            if not self.keep_backup:
                backup_parent = self.backup_path.parent
                self.backup_path.unlink(missing_ok=True)
                _prune_empty_backup_parents(backup_parent, self.backup_root)
                self.backup_path = None
        elif self.manifest_path.exists():
            self.manifest_path.unlink()
        self.committed = False

    def finalize(self, keep_backup: Optional[bool] = None) -> None:
        """
        Finalize the manifest commit after the entire logical transaction succeeds.
        Only discards `backup_path` here when `keep_backup` is False.
        """
        if not self.committed or self.finalized:
            return
        effective_keep = self.keep_backup if keep_backup is None else bool(keep_backup)
        if not effective_keep and self.backup_path is not None:
            backup_parent = self.backup_path.parent
            if self.backup_path.exists():
                assert_no_symlink_components(self.backup_path)
                self.backup_path.unlink(missing_ok=True)
            _prune_empty_backup_parents(backup_parent, self.backup_root)
            self.backup_path = None
        self.finalized = True


TransactionReceipt = Union[DatasetCommitReceipt, ManifestCommitReceipt]


def rollback_transaction_receipts(receipts: Sequence[TransactionReceipt]) -> None:
    """
    Roll back all committed directory and manifest receipts in strict reverse
    chronological order, deduplicating by object identity (`id(receipt)`).
    """
    errors: List[str] = []
    seen_ids: set[int] = set()
    unique_reversed: List[TransactionReceipt] = []
    for receipt in reversed(list(receipts)):
        rid = id(receipt)
        if rid not in seen_ids:
            seen_ids.add(rid)
            unique_reversed.append(receipt)
    for receipt in unique_reversed:
        try:
            receipt.rollback()
        except Exception as exc:
            errors.append(f"{type(receipt).__name__}: {exc}")
    if errors:
        raise RuntimeError(f"Erros durante rollback transacional: {'; '.join(errors)}")


def finalize_transaction_receipts(
    receipts: Sequence[TransactionReceipt],
    keep_backup: Optional[bool] = None,
) -> None:
    """
    Finalize all receipts after the complete logical transaction has succeeded.
    Backups are only pruned here when `keep_backup=False`.
    """
    for receipt in receipts:
        receipt.finalize(keep_backup=keep_backup)


def transactional_update_manifest(
    manifest_path: Path,
    backup_root: Path,
    write_fn: Callable[[Path], None],
    validate_fn: Callable[[Path], None],
    keep_backup: bool = True,
) -> ManifestCommitReceipt:
    """
    Transactionally update a manifest file (`.csv` or `.json`):
      1. Verify no symlinks and same filesystem between `manifest_path` and `backup_root`.
      2. Write new manifest to a sibling temporary staging file and `fsync`.
      3. Validate the staged manifest via `validate_fn`.
      4. If `manifest_path` exists, create a SHA-256 verified backup in `backup_root`.
      5. Atomically replace `manifest_path` via `Path.replace()`.
      6. Preserve backup until `finalize()` is called so multi-step transactions can roll back.
      7. On any failure, remove staging and restore the previous manifest from backup.
    """
    assert_no_symlink_components(manifest_path)
    assert_no_symlink_components(backup_root)

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    resolved_backup_root = Path(backup_root).expanduser().resolve()
    resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
    resolved_backup_root.mkdir(parents=True, exist_ok=True)

    assert_same_filesystem(resolved_manifest, resolved_backup_root)

    ts = _utc_timestamp_token()
    staging_manifest = (
        resolved_manifest.parent
        / f".staging_{resolved_manifest.name}_{ts}_{os.getpid()}.tmp"
    )

    had_previous = resolved_manifest.is_file()
    backup_path: Optional[Path] = None

    try:
        write_fn(staging_manifest)
        if not staging_manifest.is_file() or staging_manifest.stat().st_size == 0:
            raise RuntimeError(f"Manifesto em staging vazio ou não gerado: {staging_manifest}")

        with staging_manifest.open("rb") as f_sync:
            os.fsync(f_sync.fileno())

        validate_fn(staging_manifest)

        if had_previous:
            backup_dir = resolved_backup_root / ts / "manifests"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / resolved_manifest.name
            shutil.copy2(resolved_manifest, backup_path)
            if _compute_file_sha256(backup_path) != _compute_file_sha256(resolved_manifest):
                raise RuntimeError(
                    f"Falha na verificação SHA-256 do backup do manifesto: {backup_path}"
                )

        staging_manifest.replace(resolved_manifest)
    except Exception as exc:
        if staging_manifest.exists():
            staging_manifest.unlink(missing_ok=True)
        if had_previous and backup_path is not None and backup_path.is_file():
            shutil.copy2(backup_path, resolved_manifest)
            if not keep_backup:
                backup_parent = backup_path.parent
                backup_path.unlink(missing_ok=True)
                _prune_empty_backup_parents(backup_parent, resolved_backup_root)
        raise RuntimeError(
            f"Falha ao atualizar transacionalmente o manifesto '{resolved_manifest}': {exc}"
        ) from exc

    return ManifestCommitReceipt(
        manifest_path=resolved_manifest,
        backup_path=backup_path,
        had_previous_manifest=had_previous,
        keep_backup=keep_backup,
        backup_root=resolved_backup_root,
        committed=True,
        finalized=False,
    )


def transactional_write_csv_manifest(
    manifest_path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    backup_root: Path,
    key_columns: Optional[Sequence[str]] = ("dataset", "output_variant"),
    merge_existing: bool = True,
    keep_backup: bool = True,
) -> ManifestCommitReceipt:
    """
    Merge (when `merge_existing=True`) and transactionally write a CSV manifest
    with staging validation, backup, and automatic rollback.
    """
    assert_no_symlink_components(manifest_path)
    resolved_manifest = Path(manifest_path).expanduser().resolve()

    merged: Dict[tuple, Dict[str, Any]] = {}
    unkeyed_rows: List[Dict[str, Any]] = []

    if merge_existing and resolved_manifest.is_file():
        with resolved_manifest.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for existing_row in reader:
                norm_row = {k: existing_row.get(k, "") for k in fieldnames}
                if key_columns:
                    key = tuple(str(existing_row.get(col, "")) for col in key_columns)
                    merged[key] = norm_row
                else:
                    unkeyed_rows.append(norm_row)

    for row in rows:
        norm_row = {k: row.get(k, "") for k in fieldnames}
        if key_columns:
            key = tuple(str(row.get(col, "")) for col in key_columns)
            merged[key] = norm_row
        else:
            unkeyed_rows.append(norm_row)

    final_rows = list(merged.values()) if key_columns else unkeyed_rows

    def _write_csv(stage_path: Path) -> None:
        with stage_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(final_rows)

    def _validate_csv(stage_path: Path) -> None:
        with stage_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if list(reader.fieldnames or []) != list(fieldnames):
                raise ValueError(
                    f"Cabeçalho CSV inválido no staging: {reader.fieldnames} != {list(fieldnames)}"
                )
            read_rows = list(reader)
            if len(read_rows) != len(final_rows):
                raise ValueError(
                    f"Contagem de linhas CSV divergente no staging: {len(read_rows)} != {len(final_rows)}"
                )

    return transactional_update_manifest(
        manifest_path=resolved_manifest,
        backup_root=backup_root,
        write_fn=_write_csv,
        validate_fn=_validate_csv,
        keep_backup=keep_backup,
    )


def transactional_write_json_manifest(
    manifest_path: Path,
    payload: Mapping[str, Any],
    backup_root: Path,
    required_keys: Sequence[str] = ("entries",),
    keep_backup: bool = True,
) -> ManifestCommitReceipt:
    """
    Transactionally write a JSON manifest (`processing_manifest.json`) with
    staging validation, backup, and automatic rollback.
    """
    def _write_json(stage_path: Path) -> None:
        stage_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _validate_json(stage_path: Path) -> None:
        parsed = json.loads(stage_path.read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Payload JSON no staging não é um objeto/dicionário.")
        for req_key in required_keys:
            if req_key not in parsed:
                raise ValueError(f"Chave obrigatória '{req_key}' ausente no manifesto JSON em staging.")

    return transactional_update_manifest(
        manifest_path=manifest_path,
        backup_root=backup_root,
        write_fn=_write_json,
        validate_fn=_validate_json,
        keep_backup=keep_backup,
    )


__all__ = [
    "DatasetCommitReceipt",
    "ManifestCommitReceipt",
    "TransactionReceipt",
    "assert_no_symlink_components",
    "assert_no_symlinks_in_tree",
    "assert_safe_dataset_target_path",
    "assert_same_filesystem",
    "cleanup_staging_dir",
    "commit_staged_dataset_dir",
    "create_staging_dataset_dir",
    "finalize_transaction_receipts",
    "get_device_id",
    "rollback_transaction_receipts",
    "transactional_update_manifest",
    "transactional_write_csv_manifest",
    "transactional_write_json_manifest",
    "validate_staged_dataset_dir",
]

