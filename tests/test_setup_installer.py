"""Suíte de testes automatizados para o instalador reproduzível do VibNet-1D (`tests/test_setup_installer.py`).

Verifica todos os 12 critérios de aceitação da Seção 12 usando fixtures e datasets
sintéticos (sem baixar os 10 datasets reais da internet):
  1. Seleção de todos os datasets canônicos.
  2. Seleção de subconjuntos e atalhos ('cross_dataset', 'all').
  3. Rejeição de nomes inválidos com mensagem clara listando os válidos.
  4. Detecção e reutilização de download já existente (MD5 e ZIP íntegros).
  5. Retomada de download interrompido (.part) e detecção de arquivo corrompido.
  6. Processamento de dataset e reutilização quando já preparado (`REUSED_VALID`).
  7. Reprocessamento automático quando a assinatura de configuração muda ou `--force`.
  8. Detecção de arquivos/janelas ausentes ou com comprimento incorreto.
  9. Configuração de diretórios externos (`VIBNET_RAW_DIR`, `VIBNET_PROCESSED_DIR`, `VIBNET_PROCESSED_TL_DIR`, `VIBNET_RESULTS_DIR`).
  10. Recomendação de instalação em modo CPU (`--cpu-only` ou sem GPU).
  11. Detecção de GPU NVIDIA e recomendação de índice CUDA quando disponível.
  12. Geração e validação do relatório final (`installation_report.json` e `installation_report.md`).
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.setup.download_datasets import (
    compute_md5,
    download_datasets,
    download_url_with_resume,
    estimate_disk_requirements,
    resolve_selected_datasets,
    verify_existing_raw_source,
)
from scripts.setup.prepare_datasets import (
    inspect_existing_processed_variant,
    prepare_datasets,
)
from scripts.setup.system_info import (
    collect_system_info,
    recommend_pytorch_index_url,
    verify_pytorch_runtime,
)
from scripts.setup.validate_installation import validate_installation
from src.preprocessing import normalize_dataset_tl as norm_tl
from src.vibnet.data.registry import list_all_datasets
from src.vibnet.data.transactional_io import (
    assert_no_symlink_components,
    assert_no_symlinks_in_tree,
    assert_safe_dataset_target_path,
    assert_same_filesystem,
    commit_staged_dataset_dir,
    create_staging_dataset_dir,
    transactional_update_manifest,
    transactional_write_csv_manifest,
    transactional_write_json_manifest,
    validate_staged_dataset_dir,
)


def _create_synthetic_raw_fixture(raw_dir: Path, fs_cwru12k: int = 12000, fs_mfpt: int = 48828) -> None:
    """Cria fixtures sintéticas de 2 segundos para CWRU e MFPT em `raw_dir`."""
    rng = np.random.default_rng(42)

    # 1. CWRU_raw fixture (gera CWRU_12k com Class_0 e Class_1, e CWRU_48k com Class_1)
    cwru_dir = raw_dir / "CWRU_raw"
    cwru_dir.mkdir(parents=True, exist_ok=True)
    t_12k = np.linspace(0.0, 2.0, 2 * fs_cwru12k, endpoint=False)
    sig_norm = (0.08 * np.sin(2.0 * np.pi * 30.0 * t_12k) + rng.normal(0, 0.01, t_12k.shape)).astype(np.float64)
    sig_fault = (0.35 * np.sin(2.0 * np.pi * 120.0 * t_12k) + rng.normal(0, 0.02, t_12k.shape)).astype(np.float64)
    np.save(cwru_dir / "cwru12k_norm.npy", sig_norm)
    np.save(cwru_dir / "cwru12k_fault.npy", sig_fault)

    t_48k = np.linspace(0.0, 1.0, 48000, endpoint=False)
    sig_48k = (0.40 * np.sin(2.0 * np.pi * 150.0 * t_48k) + rng.normal(0, 0.02, t_48k.shape)).astype(np.float64)
    np.save(cwru_dir / "cwru48k_fault.npy", sig_48k)

    cwru_records = [
        {
            "signal_file": "cwru12k_norm.npy",
            "metainfo": {"sample_rate": 12000, "load": 0, "label": 0, "fault_size": 0.0},
        },
        {
            "signal_file": "cwru12k_fault.npy",
            "metainfo": {"sample_rate": 12000, "load": 1, "label": 1, "fault_size": 0.007},
        },
        {
            "signal_file": "cwru48k_fault.npy",
            "metainfo": {"sample_rate": 48000, "load": 0, "label": 1, "fault_size": 0.007},
        },
    ]
    (cwru_dir / "fixture_records.json").write_text(
        json.dumps(cwru_records, indent=2), encoding="utf-8"
    )

    # 2. MFPT_raw fixture (inclui sinal em 97656 Hz para testar harmonização para 48828 Hz)
    mfpt_dir = raw_dir / "MFPT_raw"
    mfpt_dir.mkdir(parents=True, exist_ok=True)
    sig_mfpt_norm_97k = (0.9 * np.sin(np.linspace(0, 50, 2 * fs_mfpt))).astype(np.float64)
    sig_mfpt_fault_48k = (2.1 * np.sin(np.linspace(0, 50, fs_mfpt))).astype(np.float64)
    np.save(mfpt_dir / "mfpt_norm.npy", sig_mfpt_norm_97k)
    np.save(mfpt_dir / "mfpt_fault.npy", sig_mfpt_fault_48k)

    mfpt_records = [
        {
            "signal_file": "mfpt_norm.npy",
            "metainfo": {"sample_rate": 97656, "load": 25, "label": 23},
        },
        {
            "signal_file": "mfpt_fault.npy",
            "metainfo": {"sample_rate": 48828, "load": 50, "label": 24},
        },
    ]
    (mfpt_dir / "fixture_records.json").write_text(
        json.dumps(mfpt_records, indent=2), encoding="utf-8"
    )


class TestSetupInstaller(unittest.TestCase):
    """Testes automatizados para os módulos de `scripts/setup/`."""

    def test_01_dataset_selection_all_and_subsets(self) -> None:
        """Critérios 1, 2 e 3: seleção de todos, subconjuntos e rejeição de nomes inválidos."""
        all_ds = resolve_selected_datasets(None)
        self.assertEqual(all_ds, list_all_datasets())
        self.assertEqual(len(all_ds), 10)

        subset = resolve_selected_datasets(["CWRU_12k", "MFPT", "CWRU_48k"])
        self.assertEqual(subset, ["CWRU_12k", "MFPT", "CWRU_48k"])

        cross_ds = resolve_selected_datasets(["cross_dataset"])
        self.assertEqual(len(cross_ds), 9)
        self.assertIn("CWRU_12k", cross_ds)
        self.assertIn("Electric_Motor", cross_ds)

        with self.assertRaises(ValueError) as ctx:
            resolve_selected_datasets(["CWRU_12k", "DATASET_INEXISTENTE"])
        self.assertIn("DATASET_INEXISTENTE", str(ctx.exception))
        self.assertIn("CWRU_12k", str(ctx.exception))

    def test_02_download_existing_interrupted_and_corrupted(self) -> None:
        """Critérios 4, 5 e 8: download existente, retomada de .part e detecção de arquivo corrompido."""
        with tempfile.TemporaryDirectory() as tmp:
            raw_dir = Path(tmp) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Cria um ZIP válido para simular MFPT.zip
            zip_bytes_io = io.BytesIO()
            with zipfile.ZipFile(zip_bytes_io, "w") as zf:
                zf.writestr("dummy.mat", b"1234567890" * 100)
            full_payload = zip_bytes_io.getvalue()
            expected_md5 = hashlib.md5(full_payload).hexdigest()

            meta = {
                "provides_datasets": ["MFPT"],
                "archive_name": "MFPT.zip",
                "expected_md5": expected_md5,
            }

            # 1) Estado ausente
            chk_missing = verify_existing_raw_source("MFPT", raw_dir, meta)
            self.assertFalse(chk_missing["valid"])
            self.assertEqual(chk_missing["status"], "MISSING")

            # 2) Simula download interrompido (.part com a primeira metade dos bytes)
            dest_zip = raw_dir / "MFPT.zip"
            part_file = raw_dir / "MFPT.zip.part"
            half = len(full_payload) // 2
            part_file.write_bytes(full_payload[:half])

            chk_part = verify_existing_raw_source("MFPT", raw_dir, meta)
            self.assertFalse(chk_part["valid"])
            self.assertEqual(chk_part["status"], "PARTIAL_DOWNLOAD_DETECTED")

            # Simula retomada HTTP 206 enviando apenas a segunda metade dos bytes
            class _FakeResponse:
                status = 206

                def __init__(self, data: bytes) -> None:
                    self._io = io.BytesIO(data)

                def read(self, n: int = -1) -> bytes:
                    return self._io.read(n)

                def __enter__(self) -> "_FakeResponse":
                    return self

                def __exit__(self, *args: object) -> None:
                    pass

            res_dl = download_url_with_resume(
                url="https://example.org/MFPT.zip",
                dest_path=dest_zip,
                expected_md5=expected_md5,
                opener=lambda req, timeout=60: _FakeResponse(full_payload[half:]),
            )
            self.assertEqual(res_dl["resumed_from_bytes"], half)
            self.assertEqual(res_dl["md5"], expected_md5)
            self.assertTrue(dest_zip.exists())
            self.assertFalse(part_file.exists())

            # 3) Verifica detecção de arquivo já existente e íntegro
            chk_ok = verify_existing_raw_source("MFPT", raw_dir, meta)
            self.assertTrue(chk_ok["valid"])
            self.assertEqual(chk_ok["status"], "ALREADY_PRESENT")

            # 4) Verifica detecção de checksum incorreto
            bad_meta = {**meta, "expected_md5": "00000000000000000000000000000000"}
            chk_bad = verify_existing_raw_source("MFPT", raw_dir, bad_meta)
            self.assertFalse(chk_bad["valid"])
            self.assertEqual(chk_bad["status"], "CHECKSUM_MISMATCH")

    def test_03_prepare_datasets_reuse_and_reprocess_on_change(self) -> None:
        """Critérios 6, 7, 8 e 9: processamento, reuso, reprocessamento e diretórios externos."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ext_raw = base / "ext_raw"
            ext_proc = base / "ext_processed"
            ext_proc_tl = base / "ext_processed_tl"

            _create_synthetic_raw_fixture(ext_raw)

            # Primeira execução: processa CWRU_12k, CWRU_48k e MFPT em todas as variantes
            manifest_1 = prepare_datasets(
                datasets=["CWRU_12k", "CWRU_48k", "MFPT"],
                variants=["all"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=False,
            )
            self.assertEqual(len(manifest_1["reused_entries"]), 0)
            for ds in ("CWRU_12k", "CWRU_48k", "MFPT"):
                for var in ("processed_1d", "raw", "detrended", "raw_rms", "detrended_rms"):
                    key = f"{ds}::{var}"
                    self.assertIn(key, manifest_1["entries"])
                    self.assertEqual(
                        manifest_1["entries"][key]["status"],
                        "PROCESSED_OK",
                        msg=f"Falha em {key}: {manifest_1['entries'][key]}",
                    )
                    self.assertGreater(manifest_1["entries"][key]["window_count"], 0)

            # Segunda execução idêntica: deve reutilizar todas as 15 combinações (REUSED_VALID)
            manifest_2 = prepare_datasets(
                datasets=["CWRU_12k", "CWRU_48k", "MFPT"],
                variants=["all"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=False,
            )
            self.assertEqual(len(manifest_2["reused_entries"]), 15)

            # Simula mudança na assinatura de configuração de MFPT::detrended no manifesto salvo
            manifest_file = ext_proc_tl / "processing_manifest.json"
            m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            m_data["entries"]["MFPT::detrended"]["config_signature"] = "outdated_sig_000"
            manifest_file.write_text(json.dumps(m_data), encoding="utf-8")

            manifest_3 = prepare_datasets(
                datasets=["MFPT"],
                variants=["detrended", "detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=False,
            )
            self.assertNotIn("MFPT::detrended", manifest_3["reused_entries"])
            self.assertEqual(manifest_3["entries"]["MFPT::detrended"]["status"], "PROCESSED_OK")
            self.assertEqual(manifest_3["entries"]["MFPT::detrended_rms"]["status"], "PROCESSED_OK")

            # Verifica detecção de janela com comprimento incorreto
            bad_dir = ext_proc / "MFPT" / "Load_25" / "Class_23"
            bad_npy = next(bad_dir.glob("*.npy"))
            np.save(bad_npy, np.zeros(123, dtype=np.float32))
            chk_corrupt = inspect_existing_processed_variant(
                "MFPT", "processed_1d", ext_proc / "MFPT", expected_window_len=48828
            )
            self.assertFalse(chk_corrupt["valid"])
            self.assertIn("Comprimento de janela divergente", chk_corrupt["reason"])

    def test_04_cpu_gpu_detection_and_final_validation_report(self) -> None:
        """Critérios 10, 11 e 12: recomendação CPU/GPU e geração do relatório de validação final."""
        # Modo sem GPU ou --cpu-only
        rec_cpu = recommend_pytorch_index_url(
            {"nvidia_smi_found": False, "gpus": []}, force_cpu=False
        )
        self.assertEqual(rec_cpu["compute_target"], "cpu")
        self.assertIn("/cpu", rec_cpu["index_url"])

        # Modo GPU NVIDIA com driver suportando CUDA 12.4
        rec_gpu = recommend_pytorch_index_url(
            {
                "nvidia_smi_found": True,
                "driver_cuda_version": "12.4",
                "gpus": [{"index": 0, "name": "NVIDIA RTX 4090", "driver_version": "550.54", "vram_mb": 24564}],
            },
            force_cpu=False,
        )
        self.assertEqual(rec_gpu["compute_target"], "cu124")
        self.assertIn("cu124", rec_gpu["index_url"])

        # Forçando --cpu-only mesmo quando há GPU
        rec_forced_cpu = recommend_pytorch_index_url(
            {
                "nvidia_smi_found": True,
                "driver_cuda_version": "12.4",
                "gpus": [{"index": 0, "name": "NVIDIA RTX 4090"}],
            },
            force_cpu=True,
        )
        self.assertEqual(rec_forced_cpu["compute_target"], "cpu")

        # Gera relatório final em diretório temporário isolado
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw_d = base / "raw"
            proc_d = base / "processed"
            proc_tl_d = base / "processed_tl"
            res_d = base / "results"
            rep_d = res_d / "setup"

            _create_synthetic_raw_fixture(raw_d)
            download_datasets(datasets=["CWRU_12k", "MFPT"], raw_dir=raw_d, dry_run=False)
            prepare_datasets(
                datasets=["CWRU_12k", "MFPT"],
                variants=["processed_1d", "detrended", "detrended_rms"],
                raw_dir=raw_d,
                processed_dir=proc_d,
                processed_tl_dir=proc_tl_d,
            )

            report = validate_installation(
                raw_dir=raw_d,
                processed_dir=proc_d,
                processed_tl_dir=proc_tl_d,
                results_dir=res_d,
                report_dir=rep_d,
                run_tests=False,
            )
            self.assertTrue((rep_d / "installation_report.json").is_file())
            self.assertTrue((rep_d / "installation_report.md").is_file())
            self.assertIn("CWRU_12k", report["datasets_inventory"]["downloaded_datasets"])
            self.assertIn("MFPT", report["datasets_inventory"]["processed_datasets_any_variant"])
            self.assertTrue(report["experiment_scripts"]["all_scripts_valid"])

    def test_05_repeated_execution_does_not_accumulate_residual_files(self) -> None:
        """Verifica que reprocessar um dataset substitui via staging+backup sem acumular arquivos residuais."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ext_raw = base / "raw"
            ext_proc = base / "processed"
            ext_proc_tl = base / "processed_tl"
            _create_synthetic_raw_fixture(ext_raw)

            prepare_datasets(
                datasets=["MFPT"],
                variants=["processed_1d", "detrended", "detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=False,
            )
            det_rms_dir = ext_proc_tl / "detrended_rms" / "MFPT"
            initial_files = sorted(
                p.relative_to(det_rms_dir).as_posix() for p in det_rms_dir.rglob("*.npy")
            )
            self.assertEqual(len(initial_files), 2)

            # Injeta 3 tipos reais de resíduos identificados na auditoria:
            # 1) Prefixo antigo s99999_w00.npy
            # 2) Pasta de classe renomeada Class_Normal/s00000_w00.npy
            # 3) Subpasta aninhada de nível 4 Class_23/Class_40/s00000_w00.npy
            residual_1 = det_rms_dir / "Load_25" / "Class_23" / "s99999_w00.npy"
            residual_2 = det_rms_dir / "Load_25" / "Class_Normal" / "s00000_w00.npy"
            residual_3 = det_rms_dir / "Load_25" / "Class_23" / "Class_40" / "s00000_w00.npy"
            for r_fp in (residual_1, residual_2, residual_3):
                r_fp.parent.mkdir(parents=True, exist_ok=True)
                np.save(r_fp, np.ones(48828, dtype=np.float64))

            self.assertEqual(len(list(det_rms_dir.rglob("*.npy"))), 5)

            # A inspeção estrutural deve reprovar a pasta contaminada devido à profundidade 4
            chk = inspect_existing_processed_variant("MFPT", "detrended_rms", det_rms_dir, 48828)
            self.assertFalse(chk["valid"])
            self.assertIn("Profundidade hierárquica inválida", chk["reason"])

            # Reprocessa MFPT::detrended_rms transacionalmente
            prepare_datasets(
                datasets=["MFPT"],
                variants=["detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=True,
            )

            final_files = sorted(
                p.relative_to(det_rms_dir).as_posix() for p in det_rms_dir.rglob("*.npy")
            )
            self.assertEqual(final_files, initial_files)
            self.assertFalse(residual_1.exists())
            self.assertFalse(residual_2.exists())
            self.assertFalse(residual_3.exists())

            # Confirma que a versão anterior contaminada foi movida para .backups/
            backups = list((ext_proc_tl / ".backups").rglob("s99999_w00.npy"))
            self.assertEqual(len(backups), 1)

    def test_06_failed_staging_and_rollback_preserves_previous_data_and_manifests(self) -> None:
        """Verifica rollback automático de diretórios e manifests em caso de falha na validação ou commit."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ext_raw = base / "raw"
            ext_proc = base / "processed"
            ext_proc_tl = base / "processed_tl"
            _create_synthetic_raw_fixture(ext_raw)

            prepare_datasets(
                datasets=["MFPT"],
                variants=["detrended", "detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
            )
            det_rms_dir = ext_proc_tl / "detrended_rms" / "MFPT"
            norm_manifest = ext_proc_tl / "detrended_rms" / "normalization_manifest.csv"

            before_hashes = {
                p.relative_to(det_rms_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in det_rms_dir.rglob("*.npy")
            }
            manifest_before_sha = hashlib.sha256(norm_manifest.read_bytes()).hexdigest()

            # Caso A: Corrompe um arquivo na entrada detrended com NaN para forçar falha no staging
            corrupt_input = next((ext_proc_tl / "detrended" / "MFPT").rglob("*.npy"))
            valid_input_bytes = corrupt_input.read_bytes()
            np.save(corrupt_input, np.full(48828, np.nan, dtype=np.float64))

            res_failed = prepare_datasets(
                datasets=["MFPT"],
                variants=["detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=True,
            )
            self.assertEqual(res_failed["entries"]["MFPT::detrended_rms"]["status"], "NORMALIZATION_FAILED")

            # O diretório detrended_rms/MFPT e o normalization_manifest.csv devem permanecer 100% intactos
            after_hashes = {
                p.relative_to(det_rms_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in det_rms_dir.rglob("*.npy")
            }
            self.assertEqual(before_hashes, after_hashes)
            self.assertEqual(
                hashlib.sha256(norm_manifest.read_bytes()).hexdigest(),
                manifest_before_sha,
            )
            # Nenhum diretório .staging_* deve ter sobrado
            leftover_staging = list((ext_proc_tl / "detrended_rms").glob(".staging_*"))
            self.assertEqual(len(leftover_staging), 0)

            # Restaura o arquivo de entrada e testa Caso B: rollback explícito de DatasetCommitReceipt e ManifestCommitReceipt
            corrupt_input.write_bytes(valid_input_bytes)
            v_root = ext_proc_tl / "detrended_rms"
            b_root = ext_proc_tl / ".backups"
            st_dir = create_staging_dataset_dir(v_root, "MFPT", backup_root=b_root)
            for rel_p in before_hashes:
                out_f = st_dir / rel_p
                out_f.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_f, np.full(48828, 7.0, dtype=np.float64))

            ds_receipt = commit_staged_dataset_dir(
                staging_dir=st_dir,
                target_dir=det_rms_dir,
                allowed_variant_root=v_root,
                backup_root=b_root,
                dataset_name="MFPT",
                variant_name="detrended_rms",
            )
            # Confirma que os arquivos mudaram após o commit
            committed_hashes = {
                p.relative_to(det_rms_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in det_rms_dir.rglob("*.npy")
            }
            self.assertNotEqual(before_hashes, committed_hashes)

            # Executa rollback do receipt e verifica restauração bit-a-bit do estado original
            ds_receipt.rollback()
            restored_hashes = {
                p.relative_to(det_rms_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in det_rms_dir.rglob("*.npy")
            }
            self.assertEqual(before_hashes, restored_hashes)

    def test_07_exact_relative_paths_symlinks_same_fs_and_transactional_manifests(self) -> None:
        """Verifica os 4 ajustes técnicos obrigatórios: caminhos relativos 1:1, anti-symlink, mesmo FS e manifests transacionais."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src_dir = base / "source_ds"
            v_root = base / "detrended_rms"
            b_root = base / ".backups"
            v_root.mkdir(parents=True, exist_ok=True)

            # 1. Ajuste 1: mesma contagem (2 arquivos), mas caminho relativo divergente -> deve falhar
            f1 = src_dir / "Load_0" / "Class_0" / "s00000_w00.npy"
            f2 = src_dir / "Load_0" / "Class_1" / "s00001_w00.npy"
            for f in (f1, f2):
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, np.ones(128, dtype=np.float64))

            st_dir = create_staging_dataset_dir(v_root, "CWRU_12k", backup_root=b_root)
            s1 = st_dir / "Load_0" / "Class_0" / "s00000_w00.npy"
            # Caminho relativo trocado (s00099_w00.npy em vez de s00001_w00.npy)
            s2_wrong = st_dir / "Load_0" / "Class_1" / "s00099_w00.npy"
            for s in (s1, s2_wrong):
                s.parent.mkdir(parents=True, exist_ok=True)
                np.save(s, np.ones(128, dtype=np.float64))

            val_mismatch = validate_staged_dataset_dir(
                staging_dir=st_dir,
                expected_window_count=2,
                expected_window_len=128,
                reference_rms=1.0,
                source_dir=src_dir,
            )
            self.assertFalse(val_mismatch["valid"])
            self.assertIn("Caminhos relativos divergentes", val_mismatch["reason"])
            self.assertEqual(val_mismatch["missing_relative_paths"], ["Load_0/Class_1/s00001_w00.npy"])
            self.assertEqual(val_mismatch["extra_relative_paths"], ["Load_0/Class_1/s00099_w00.npy"])

            # Corrige o caminho relativo -> deve aprovar
            s2_wrong.replace(st_dir / "Load_0" / "Class_1" / "s00001_w00.npy")
            val_ok = validate_staged_dataset_dir(
                staging_dir=st_dir,
                expected_window_count=2,
                expected_window_len=128,
                reference_rms=1.0,
                source_dir=src_dir,
            )
            self.assertTrue(val_ok["valid"])

            # 2. Ajuste 3a: Proteção contra links simbólicos
            sym_file = st_dir / "Load_0" / "Class_0" / "sym_w01.npy"
            sym_file.symlink_to(f1)
            val_sym = validate_staged_dataset_dir(staging_dir=st_dir)
            self.assertFalse(val_sym["valid"])
            self.assertIn("links simbólicos", val_sym["reason"])
            with self.assertRaises(PermissionError):
                assert_no_symlinks_in_tree(st_dir)
            sym_file.unlink()

            # Proteção contra path traversal / dataset fora da lista
            with self.assertRaises(ValueError):
                assert_safe_dataset_target_path(v_root / "../escape", v_root, "../escape")

            # 3. Ajuste 3b: Mesmo filesystem (st_dev)
            dev_id = assert_same_filesystem(st_dir, v_root, b_root)
            self.assertIsInstance(dev_id, int)

            # 4. Ajuste 4: Manifesto transacional com backup e rollback em falha de validação
            csv_manifest = v_root / "normalization_manifest.csv"
            fields = ["dataset", "output_variant", "input_windows"]
            m_rec1 = transactional_write_csv_manifest(
                manifest_path=csv_manifest,
                fieldnames=fields,
                rows=[{"dataset": "HUST", "output_variant": "detrended_rms", "input_windows": 951}],
                backup_root=b_root,
            )
            self.assertTrue(csv_manifest.is_file())
            content_v1 = csv_manifest.read_text(encoding="utf-8")
            self.assertIn("951", content_v1)

            # Falha proposital na validação de uma atualização subsequente -> deve preservar content_v1
            with self.assertRaises(RuntimeError):
                transactional_update_manifest(
                    manifest_path=csv_manifest,
                    backup_root=b_root,
                    write_fn=lambda p: p.write_text("corrupted_content", encoding="utf-8"),
                    validate_fn=lambda p: (_ for _ in ()).throw(ValueError("Cabeçalho inválido")),
                )
            self.assertEqual(csv_manifest.read_text(encoding="utf-8"), content_v1)

            # Atualização válida seguida de rollback explícito do receipt
            m_rec2 = transactional_write_csv_manifest(
                manifest_path=csv_manifest,
                fieldnames=fields,
                rows=[{"dataset": "HUST", "output_variant": "detrended_rms", "input_windows": 999}],
                backup_root=b_root,
            )
            self.assertIn("999", csv_manifest.read_text(encoding="utf-8"))
            m_rec2.rollback()
            self.assertEqual(csv_manifest.read_text(encoding="utf-8"), content_v1)

    def test_08_coordinated_rollback_across_dirs_normalization_and_processing_manifests(self) -> None:
        """
        Verifica a transação lógica única (diretórios processados + normalization_manifest.csv +
        processing_manifest.json) com restauração bit-a-bit em falha de processing_manifest.json,
        tanto com keep_backup=True quanto com keep_backup=False, e limpeza diferida de backups.
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ext_raw = base / "raw"
            ext_proc = base / "processed"
            ext_proc_tl = base / "processed_tl"
            _create_synthetic_raw_fixture(ext_raw)

            # 1. Execução inicial válida: cria detrended/MFPT, detrended_rms/MFPT,
            #    normalization_manifest.csv e processing_manifest.json (Estado V1).
            prepare_datasets(
                datasets=["MFPT"],
                variants=["detrended", "detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=False,
                keep_backup=True,
            )

            det_dir = ext_proc_tl / "detrended" / "MFPT"
            det_rms_dir = ext_proc_tl / "detrended_rms" / "MFPT"
            norm_manifest_path = ext_proc_tl / "detrended_rms" / "normalization_manifest.csv"
            proc_manifest_path = ext_proc_tl / "processing_manifest.json"

            self.assertTrue(det_dir.is_dir())
            self.assertTrue(det_rms_dir.is_dir())
            self.assertTrue(norm_manifest_path.is_file())
            self.assertTrue(proc_manifest_path.is_file())

            def _snapshot_tree_hashes(root: Path) -> dict[str, str]:
                return {
                    p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.rglob("*.npy"))
                }

            v1_det_hashes = _snapshot_tree_hashes(det_dir)
            v1_det_rms_hashes = _snapshot_tree_hashes(det_rms_dir)
            v1_norm_manifest_bytes = norm_manifest_path.read_bytes()
            v1_norm_manifest_sha = hashlib.sha256(v1_norm_manifest_bytes).hexdigest()
            v1_proc_manifest_bytes = proc_manifest_path.read_bytes()
            v1_proc_manifest_sha = hashlib.sha256(v1_proc_manifest_bytes).hexdigest()

            # 2. Altera o sinal bruto sintético do MFPT para que o próximo processamento (V2)
            #    gere janelas .npy distintas e um novo RMS_ref em normalization_manifest.csv.
            mfpt_raw_dir = ext_raw / "MFPT_raw"
            for raw_npy in mfpt_raw_dir.glob("*.npy"):
                arr = np.load(raw_npy, allow_pickle=False)
                np.save(raw_npy, arr * 3.5 + 0.125)

            # 3. Simula falha na atualização final de processing_manifest.json DEPOIS que
            #    detrended/MFPT, detrended_rms/MFPT e normalization_manifest.csv já foram
            #    promovidos para o disco, primeiro com keep_backup=True e depois com keep_backup=False.
            observed_mid_transaction: list[dict[str, bool]] = []

            for keep_bkp in (True, False):
                def _failing_write_json_manifest(*args, **kwargs):
                    # No exato momento em que processing_manifest.json vai ser escrito,
                    # os diretórios e o normalization_manifest.csv já foram atualizados para V2!
                    current_det_hashes = _snapshot_tree_hashes(det_dir)
                    current_det_rms_hashes = _snapshot_tree_hashes(det_rms_dir)
                    current_norm_sha = hashlib.sha256(norm_manifest_path.read_bytes()).hexdigest()
                    observed_mid_transaction.append(
                        {
                            "keep_backup": keep_bkp,
                            "det_changed_before_fail": current_det_hashes != v1_det_hashes,
                            "det_rms_changed_before_fail": current_det_rms_hashes != v1_det_rms_hashes,
                            "norm_manifest_changed_before_fail": current_norm_sha != v1_norm_manifest_sha,
                        }
                    )
                    raise RuntimeError(
                        f"Falha simulada ao gravar processing_manifest.json (keep_backup={keep_bkp})"
                    )

                with patch(
                    "scripts.setup.prepare_datasets.transactional_write_json_manifest",
                    side_effect=_failing_write_json_manifest,
                ):
                    with self.assertRaises(RuntimeError):
                        prepare_datasets(
                            datasets=["MFPT"],
                            variants=["detrended", "detrended_rms"],
                            raw_dir=ext_raw,
                            processed_dir=ext_proc,
                            processed_tl_dir=ext_proc_tl,
                            force=True,
                            keep_backup=keep_bkp,
                        )

                # Confirma que os diretórios e normalization_manifest.csv realmente chegaram a mudar
                # para V2 antes da falha simulada em processing_manifest.json
                self.assertTrue(observed_mid_transaction[-1]["det_changed_before_fail"])
                self.assertTrue(observed_mid_transaction[-1]["det_rms_changed_before_fail"])
                self.assertTrue(observed_mid_transaction[-1]["norm_manifest_changed_before_fail"])

                # 3, 4 e 5. Verifica restauração bit-a-bit dos diretórios originais,
                # de normalization_manifest.csv e de processing_manifest.json
                self.assertEqual(
                    _snapshot_tree_hashes(det_dir),
                    v1_det_hashes,
                    f"detrended/MFPT não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                )
                self.assertEqual(
                    _snapshot_tree_hashes(det_rms_dir),
                    v1_det_rms_hashes,
                    f"detrended_rms/MFPT não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                )
                self.assertEqual(
                    hashlib.sha256(norm_manifest_path.read_bytes()).hexdigest(),
                    v1_norm_manifest_sha,
                    f"normalization_manifest.csv não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                )
                self.assertEqual(
                    norm_manifest_path.read_bytes(),
                    v1_norm_manifest_bytes,
                )
                self.assertEqual(
                    hashlib.sha256(proc_manifest_path.read_bytes()).hexdigest(),
                    v1_proc_manifest_sha,
                    f"processing_manifest.json não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                )
                self.assertEqual(
                    proc_manifest_path.read_bytes(),
                    v1_proc_manifest_bytes,
                )

                # 7. Verifica ausência total de diretórios/arquivos de staging residuais
                staging_residuals = [
                    p
                    for root_d in (ext_proc, ext_proc_tl)
                    if root_d.exists()
                    for p in root_d.rglob(".staging_*")
                ]
                self.assertEqual(
                    staging_residuals,
                    [],
                    f"Resíduos de staging encontrados (keep_backup={keep_bkp}): {staging_residuals}",
                )

            # 6b. Agora executa uma transação completa de sucesso com keep_backup=False em um backup_dir limpo
            #     e verifica que V2 é confirmado e que os backups descartáveis são removidos somente no final.
            clean_bkp_dir = base / "clean_backups_no_keep"
            prepare_datasets(
                datasets=["MFPT"],
                variants=["detrended", "detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=True,
                backup_dir=clean_bkp_dir,
                keep_backup=False,
            )
            self.assertNotEqual(_snapshot_tree_hashes(det_dir), v1_det_hashes)
            self.assertNotEqual(_snapshot_tree_hashes(det_rms_dir), v1_det_rms_hashes)
            self.assertNotEqual(
                hashlib.sha256(norm_manifest_path.read_bytes()).hexdigest(),
                v1_norm_manifest_sha,
            )
            remaining_backup_files = (
                [p for p in clean_bkp_dir.rglob("*") if p.is_file()]
                if clean_bkp_dir.exists()
                else []
            )
            self.assertEqual(
                remaining_backup_files,
                [],
                f"Backups não foram limpos após finalização com keep_backup=False: {remaining_backup_files}",
            )

    def test_09_normalize_dataset_tl_cross_variant_rollback_raw_and_detrended(self) -> None:
        """
        Verifica que em normalize_dataset_tl.main(), uma falha na gravação do manifesto
        da segunda variante (detrended) reverte todos os recibos já confirmados na execução:
          - Diretórios da variante raw (raw_rms/MFPT);
          - Manifesto raw (raw_rms/normalization_manifest.csv);
          - Diretórios da variante detrended (detrended_rms/MFPT);
          - Manifesto detrended (detrended_rms/normalization_manifest.csv),
        tanto com keep_backup=True quanto com keep_backup=False (--no-backup).
        """
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ext_raw = base / "raw"
            ext_proc = base / "processed"
            ext_proc_tl = base / "processed_tl"
            _create_synthetic_raw_fixture(ext_raw)

            # 1. Estado inicial V1: prepara variantes raw, detrended, raw_rms e detrended_rms para MFPT
            prepare_datasets(
                datasets=["MFPT"],
                variants=["raw", "detrended", "raw_rms", "detrended_rms"],
                raw_dir=ext_raw,
                processed_dir=ext_proc,
                processed_tl_dir=ext_proc_tl,
                force=False,
                keep_backup=True,
            )

            raw_in_dir = ext_proc_tl / "raw" / "MFPT"
            det_in_dir = ext_proc_tl / "detrended" / "MFPT"
            raw_rms_dir = ext_proc_tl / "raw_rms" / "MFPT"
            det_rms_dir = ext_proc_tl / "detrended_rms" / "MFPT"
            raw_manifest_path = ext_proc_tl / "raw_rms" / "normalization_manifest.csv"
            det_manifest_path = ext_proc_tl / "detrended_rms" / "normalization_manifest.csv"

            def _snapshot_tree_hashes(root: Path) -> dict[str, str]:
                return {
                    p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.rglob("*.npy"))
                }

            v1_raw_rms_hashes = _snapshot_tree_hashes(raw_rms_dir)
            v1_det_rms_hashes = _snapshot_tree_hashes(det_rms_dir)
            v1_raw_manifest_bytes = raw_manifest_path.read_bytes()
            v1_raw_manifest_sha = hashlib.sha256(v1_raw_manifest_bytes).hexdigest()
            v1_det_manifest_bytes = det_manifest_path.read_bytes()
            v1_det_manifest_sha = hashlib.sha256(v1_det_manifest_bytes).hexdigest()

            # 2. Altera as janelas de entrada em raw/MFPT e detrended/MFPT (Estado V2)
            #    para que a nova execução de normalize_dataset_tl gere arquivos e manifests diferentes.
            for in_root in (raw_in_dir, det_in_dir):
                for npy_f in sorted(in_root.rglob("*.npy")):
                    arr = np.load(npy_f, allow_pickle=False)
                    # Escala e adiciona leve curvatura não-linear para mudar tanto o RMS_ref quanto o sinal normalizado
                    np.save(npy_f, arr * 2.75 + 0.05 * np.square(arr))

            orig_input_root = norm_tl.INPUT_ROOT
            orig_fallback_root = norm_tl.PROCESSED_FALLBACK_ROOT
            orig_variants = dict(norm_tl.VARIANTS)

            try:
                norm_tl.INPUT_ROOT = ext_proc_tl
                norm_tl.PROCESSED_FALLBACK_ROOT = ext_proc
                norm_tl.VARIANTS = {
                    "raw": {
                        "input_dir": ext_proc_tl / "raw",
                        "output_dir": ext_proc_tl / "raw_rms",
                    },
                    "detrended": {
                        "input_dir": ext_proc_tl / "detrended",
                        "output_dir": ext_proc_tl / "detrended_rms",
                    },
                }

                real_write_manifest = norm_tl.write_manifest

                for keep_bkp in (True, False):
                    mid_state: dict[str, bool] = {}

                    def _write_manifest_fail_on_detrended(
                        rows,
                        output_root,
                        merge_existing=True,
                        backup_root=None,
                        keep_backup=True,
                        return_receipt=False,
                    ):
                        # Permite que o manifesto raw seja gravado e confirmado normalmente
                        if Path(output_root).name == "raw_rms":
                            return real_write_manifest(
                                rows,
                                output_root,
                                merge_existing=merge_existing,
                                backup_root=backup_root,
                                keep_backup=keep_backup,
                                return_receipt=return_receipt,
                            )
                        # No momento de gravar o manifesto detrended, verifica que raw_rms/MFPT,
                        # raw_rms/normalization_manifest.csv e detrended_rms/MFPT já foram promovidos para V2!
                        mid_state["raw_rms_changed"] = _snapshot_tree_hashes(raw_rms_dir) != v1_raw_rms_hashes
                        mid_state["raw_manifest_changed"] = (
                            hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest() != v1_raw_manifest_sha
                        )
                        mid_state["det_rms_changed"] = _snapshot_tree_hashes(det_rms_dir) != v1_det_rms_hashes
                        raise RuntimeError(
                            f"Falha simulada ao gravar normalization_manifest.csv de detrended_rms (keep_backup={keep_bkp})"
                        )

                    cli_args = [
                        "--datasets",
                        "MFPT",
                        "--variants",
                        "raw",
                        "detrended",
                        "--backup-dir",
                        str(ext_proc_tl / ".backups"),
                    ]
                    if not keep_bkp:
                        cli_args.append("--no-backup")

                    with patch.object(norm_tl, "write_manifest", side_effect=_write_manifest_fail_on_detrended):
                        with self.assertRaises(RuntimeError):
                            norm_tl.main(cli_args)

                    # Confirma que os commits de raw_rms, manifesto raw e detrended_rms ocorreram antes da falha
                    self.assertTrue(mid_state.get("raw_rms_changed"))
                    self.assertTrue(mid_state.get("raw_manifest_changed"))
                    self.assertTrue(mid_state.get("det_rms_changed"))

                    # 4 e 5. Verifica restauração bit a bit de raw_rms/MFPT, manifesto raw,
                    # detrended_rms/MFPT e manifesto detrended
                    self.assertEqual(
                        _snapshot_tree_hashes(raw_rms_dir),
                        v1_raw_rms_hashes,
                        f"raw_rms/MFPT não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                    )
                    self.assertEqual(
                        hashlib.sha256(raw_manifest_path.read_bytes()).hexdigest(),
                        v1_raw_manifest_sha,
                        f"Manifesto raw não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                    )
                    self.assertEqual(raw_manifest_path.read_bytes(), v1_raw_manifest_bytes)

                    self.assertEqual(
                        _snapshot_tree_hashes(det_rms_dir),
                        v1_det_rms_hashes,
                        f"detrended_rms/MFPT não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                    )
                    self.assertEqual(
                        hashlib.sha256(det_manifest_path.read_bytes()).hexdigest(),
                        v1_det_manifest_sha,
                        f"Manifesto detrended não foi restaurado bit-a-bit (keep_backup={keep_bkp})",
                    )
                    self.assertEqual(det_manifest_path.read_bytes(), v1_det_manifest_bytes)

                    # Confirma ausência de resíduos .staging_*
                    staging_residuals = list(ext_proc_tl.rglob(".staging_*"))
                    self.assertEqual(staging_residuals, [])
            finally:
                norm_tl.INPUT_ROOT = orig_input_root
                norm_tl.PROCESSED_FALLBACK_ROOT = orig_fallback_root
                norm_tl.VARIANTS = orig_variants


if __name__ == "__main__":
    unittest.main()
