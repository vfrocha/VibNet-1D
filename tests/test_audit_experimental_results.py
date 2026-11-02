#!/usr/bin/env python3
"""
Validation and unit tests for scripts/audit_experimental_results.py.
Validates the generated audit outputs against a sample of original CSVs in results/
and tests edge cases (shifted columns, non-OK status, unrecognized columns).
Compatible with pure Python standard library.
"""
import csv
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "audit_experimental_results.py"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "reports" / "reproducibility_audit"


def load_csv_dicts(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class TestAuditExperimentalResultsValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the audit report is freshly generated before running validation checks
        proc = subprocess.run(
            [sys.executable, str(AUDIT_SCRIPT), "--results-dir", str(RESULTS_DIR), "--output-dir", str(REPORT_DIR)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        cls.cli_output = proc.stdout
        cls.inventory = load_csv_dicts(REPORT_DIR / "results_inventory.csv")
        cls.summary = load_csv_dicts(REPORT_DIR / "metrics_summary.csv")
        cls.anomalies = load_csv_dicts(REPORT_DIR / "anomalies_and_warnings.csv")

    def test_inventory_completeness_and_zero_errors(self):
        self.assertEqual(len(self.inventory), 220)
        csv_items = [r for r in self.inventory if r["type"] == ".csv"]
        self.assertEqual(len(csv_items), 108)
        errors = [r for r in self.inventory if r["status"] == "error"]
        self.assertEqual(len(errors), 0, f"Unexpected read errors: {errors}")

        granularities = {r["granularity"] for r in csv_items}
        self.assertEqual(granularities, {"fold_level", "summary", "dataset_audit"})

    def test_sample_validation_against_original_csvs(self):
        """Validate metrics_summary.csv against direct calculation on original CSVs."""
        # 1. Check tl_four_variants detrended_rms MLP
        orig_fv = load_csv_dicts(RESULTS_DIR / "tl_four_variants/run_20261005_163159/results.csv")
        fv_vals = [
            float(r["Macro F1"])
            for r in orig_fv
            if r["Target Dataset"] == "CWRU_48k"
            and r["Model"] == "MLP"
            and r["Variant"] == "detrended_rms"
        ]
        self.assertEqual(len(fv_vals), 12)
        fv_sum = [
            r for r in self.summary
            if r["relative_path"] == "tl_four_variants/run_20261005_163159/results.csv"
            and r["dataset"] == "CWRU_48k"
            and r["model"] == "MLP"
            and r["strategy"] == "detrended_rms"
            and r["metric"] == "macro_f1"
        ]
        self.assertEqual(len(fv_sum), 1)
        self.assertEqual(int(fv_sum[0]["count"]), 12)
        self.assertAlmostEqual(float(fv_sum[0]["mean"]), statistics.mean(fv_vals), places=7)
        self.assertAlmostEqual(float(fv_sum[0]["std"]), statistics.stdev(fv_vals), places=7)
        self.assertAlmostEqual(float(fv_sum[0]["mean"]), 0.92020318, places=6)

        # 2. Check tl_balance_strategies smote detrended_rms MLP
        orig_bs = load_csv_dicts(RESULTS_DIR / "tl_balance_strategies/run_20261012_150540/results.csv")
        bs_vals = [
            float(r["Macro F1"])
            for r in orig_bs
            if r["Target Dataset"] == "CWRU_48k"
            and r["Model"] == "MLP"
            and r["Variant"] == "detrended_rms"
        ]
        self.assertEqual(len(bs_vals), 12)
        bs_sum = [
            r for r in self.summary
            if r["relative_path"] == "tl_balance_strategies/run_20261012_150540/results.csv"
            and r["dataset"] == "CWRU_48k"
            and r["model"] == "MLP"
            and r["metric"] == "macro_f1"
        ]
        self.assertEqual(len(bs_sum), 1)
        self.assertAlmostEqual(float(bs_sum[0]["mean"]), statistics.mean(bs_vals), places=7)
        self.assertAlmostEqual(float(bs_sum[0]["mean"]), 0.93888498, places=6)

        # 3. Check tl_cross_dataset_mlp_tabnet UORED|Mechanical_Gear|Electric_Motor -> CWRU_48k MLP (36 fold evaluations)
        orig_cd = load_csv_dicts(RESULTS_DIR / "tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv")
        cd_vals = [
            float(r["Macro F1"])
            for r in orig_cd
            if r["Target"] == "CWRU_48k"
            and r["Model"] == "MLP"
            and r["Source_Combo"] == "UORED|Mechanical_Gear|Electric_Motor"
            and r["Fold_Status"] == "OK"
        ]
        self.assertEqual(len(cd_vals), 36)
        cd_sum = [
            r for r in self.summary
            if r["relative_path"] == "tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv"
            and r["dataset"] == "CWRU_48k"
            and r["model"] == "MLP"
            and r["sources"] == "UORED|Mechanical_Gear|Electric_Motor"
            and r["metric"] == "macro_f1"
        ]
        self.assertEqual(len(cd_sum), 1)
        self.assertEqual(int(cd_sum[0]["count"]), 36)
        self.assertEqual(int(cd_sum[0]["n_seeds"]), 3)
        self.assertEqual(int(cd_sum[0]["n_folds"]), 12)
        self.assertAlmostEqual(float(cd_sum[0]["mean"]), statistics.mean(cd_vals), places=7)
        self.assertAlmostEqual(float(cd_sum[0]["mean"]), 0.92318622, places=6)

    def test_anomaly_detection_shifted_columns_and_tabnet_errors(self):
        shifted = [a for a in self.anomalies if a["anomaly_type"] == "SHIFTED_COLUMN_CORRUPTION"]
        self.assertEqual(len(shifted), 100)

        non_ok = [a for a in self.anomalies if a["anomaly_type"] == "NON_OK_STATUS"]
        self.assertEqual(len(non_ok), 60)
        for a in non_ok:
            self.assertEqual(a["dataset"], "PU")
            self.assertEqual(a["model"], "TabNet")
            self.assertEqual(a["fold"], "C4_1500rpm_0.7Nm_400N")

        # Verify that shifted MLP (Scratch) rows in baseline_loco_results_20261003_052352.csv
        # are marked invalid and not averaged as valid macro_f1
        corrupted_mlp = [
            r for r in self.summary
            if r["relative_path"] == "baseline_loco_results_20261003_052352.csv"
            and r["model"] == "MLP (Scratch)"
            and r["metric"] == "macro_f1"
        ]
        for r in corrupted_mlp:
            self.assertEqual(int(r["count"]), 0)
            self.assertGreater(int(r["invalid_count"]), 0)

    def test_synthetic_edge_cases_in_tempdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            res_dir = tmp_path / "results"
            out_dir = tmp_path / "out"
            res_dir.mkdir()
            sample_csv = res_dir / "custom_exp.csv"
            sample_csv.write_text(
                "Dataset,Model,Test Condition,Seed,Custom_Unknown_Col,Macro F1,Status\n"
                "CWRU_48k,MLP,Fold_1,42,foo,0.95,OK\n"
                "CWRU_48k,MLP,Fold_2,42,bar,0.85,ERROR_FIT\n",
                encoding="utf-8",
            )
            subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT), "--results-dir", str(res_dir), "--output-dir", str(out_dir)],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
            )
            inv = load_csv_dicts(out_dir / "results_inventory.csv")
            self.assertEqual(len(inv), 1)
            self.assertEqual(inv[0]["unrecognized_columns"], "Custom_Unknown_Col")

            summ = load_csv_dicts(out_dir / "metrics_summary.csv")
            self.assertEqual(len(summ), 1)
            self.assertEqual(int(summ[0]["count"]), 1)
            self.assertEqual(int(summ[0]["invalid_count"]), 1)
            self.assertAlmostEqual(float(summ[0]["mean"]), 0.95, places=6)


if __name__ == "__main__":
    unittest.main()
