#!/usr/bin/env python3
"""
Consolidated Statistical Comparison: Classical ML Baselines vs. Neural Scratch vs. Transfer Learning
(`src/experiments/analyze_scratch_tl_classical_comparison.py`).

Combines in read-only mode:
  1. Existing `MLP` and `TabNet` (`Scratch`, `Fine-Tuning`, `Linear Probing`) results from
     `results/scratch_vs_tl/run_20261105_141743/`.
  2. Complementary `RandomForest`, `SVM`, and `XGBoost` (`ClassicalBaseline`) results from
     `results/scratch_vs_tl/classical_baselines/run_<timestamp>/`.

Methodological Guarantees (Section 9):
  - Does NOT modify any files in `run_20261105_141743` or `classical_baselines`.
  - Verifies matched `(target_dataset, fold_id, condition, seed, balance_strategy)` keys
    and asserts 100% parity of `split_signature` and `preprocessing_signature` across all
    paired evaluations.
  - Never treats different seeds, folds, or balance strategies as independent datasets:
    1. Within-dataset inference uses the LOCO fold (`N = K_T` folds per dataset) as the
       statistical unit after averaging across the 3 seeds within each fold, applying
       two-sided Wilcoxon signed-rank tests with Holm-Bonferroni step-down correction
       separately per `(target_dataset, balance_strategy, metric)`.
    2. Cross-dataset generalization analysis stays strictly at the dataset level
       (`N = 5` target datasets, separately per `balance_strategy`), computing per-dataset
       rankings, mean ranks across the 5 datasets, and Friedman omnibus tests, clearly
       distinguishing descriptive rankings from statistically significant differences.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config import RESULTS_DIR

DEFAULT_REFERENCE_BENCHMARK_DIR: Path = (
    RESULTS_DIR / "scratch_vs_tl" / "run_20261105_141743"
)
DEFAULT_CLASSICAL_ROOT: Path = RESULTS_DIR / "scratch_vs_tl" / "classical_baselines"
DEFAULT_CONSOLIDATED_ROOT: Path = RESULTS_DIR / "scratch_vs_tl" / "consolidated_comparison"
METRICS_COLS: List[str] = ["macro_f1", "balanced_accuracy", "accuracy", "roc_auc"]


def _holm_bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down adjustment to a family of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(
        enumerate(p_values),
        key=lambda x: (math.isnan(x[1]), x[1] if not math.isnan(x[1]) else 1.0),
    )
    adjusted = [float("nan")] * m
    running_max = 0.0
    valid_indices = [i for i, p in indexed if not math.isnan(p)]
    k_valid = len(valid_indices)
    for rank_idx, orig_idx in enumerate(valid_indices):
        raw_p = float(p_values[orig_idx])
        adj = min(1.0, raw_p * (k_valid - rank_idx))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def _df_to_table_str(df: pd.DataFrame) -> str:
    """Format DataFrame as a clean text table for Markdown reports."""
    if df.empty:
        return "(Nenhuma linha disponível)"
    return df.to_string(index=False)


def resolve_latest_classical_run_dir(
    classical_root: str | Path = DEFAULT_CLASSICAL_ROOT,
) -> Path:
    """Find the most recent `run_*` directory under `results/scratch_vs_tl/classical_baselines/`."""
    root = Path(classical_root)
    if not root.exists():
        raise FileNotFoundError(
            f"Classical baselines directory '{root}' does not exist yet."
        )
    candidates = sorted(
        [
            d
            for d in root.iterdir()
            if d.is_dir() and d.name.startswith("run_") and (d / "results.csv").exists()
        ],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed classical baseline run directories containing results.csv found in '{root}'."
        )
    return candidates[-1]


def verify_and_merge_matched_results(
    df_neural: pd.DataFrame,
    df_classical: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Verify that `df_neural` and `df_classical` share identical `split_signature` and
    `preprocessing_signature` on every matched `(target_dataset, fold_id, condition, seed, balance_strategy)`
    block, and return the merged dataframe on the matched blocks.
    """
    match_cols = ["target_dataset", "fold_id", "condition", "seed", "balance_strategy"]

    neural_ok = df_neural[df_neural["status"] == "OK"].copy()
    classical_ok = df_classical[df_classical["status"] == "OK"].copy()

    if neural_ok.empty or classical_ok.empty:
        raise ValueError(
            f"Cannot consolidate empty results: neural_ok={len(neural_ok)}, "
            f"classical_ok={len(classical_ok)}."
        )

    neural_blocks = (
        neural_ok[match_cols + ["split_signature", "preprocessing_signature"]]
        .drop_duplicates()
        .set_index(match_cols)
    )
    classical_blocks = (
        classical_ok[match_cols + ["split_signature", "preprocessing_signature"]]
        .drop_duplicates()
        .set_index(match_cols)
    )

    common_keys = sorted(
        set(neural_blocks.index.tolist()).intersection(
            set(classical_blocks.index.tolist())
        )
    )
    if not common_keys:
        raise ValueError(
            "Zero overlapping (target_dataset, fold_id, condition, seed, balance_strategy) "
            "blocks found between neural and classical results!"
        )

    for key in common_keys:
        n_split = str(neural_blocks.loc[key, "split_signature"])
        c_split = str(classical_blocks.loc[key, "split_signature"])
        if n_split != c_split:
            raise ValueError(
                f"split_signature mismatch on {key}: neural='{n_split}' vs classical='{c_split}'"
            )
        n_prep = str(neural_blocks.loc[key, "preprocessing_signature"])
        c_prep = str(classical_blocks.loc[key, "preprocessing_signature"])
        if n_prep != c_prep:
            raise ValueError(
                f"preprocessing_signature mismatch on {key}: neural='{n_prep}' vs classical='{c_prep}'"
            )

    common_index = pd.MultiIndex.from_tuples(common_keys, names=match_cols)
    n_filtered = neural_ok[
        neural_ok.set_index(match_cols).index.isin(common_index)
    ].copy()
    c_filtered = classical_ok[
        classical_ok.set_index(match_cols).index.isin(common_index)
    ].copy()

    n_filtered["model_family"] = "Neural_DL"
    c_filtered["model_family"] = "Classical_ML"

    merged = pd.concat([n_filtered, c_filtered], ignore_index=True)
    merged["model_config"] = (
        merged["architecture"].astype(str) + " (" + merged["method"].astype(str) + ")"
    )

    audit_info = {
        "neural_total_ok_rows": int(len(neural_ok)),
        "classical_total_ok_rows": int(len(classical_ok)),
        "matched_fold_seed_balance_blocks": int(len(common_keys)),
        "merged_total_rows": int(len(merged)),
        "signatures_verified_identical": True,
        "target_datasets_covered": sorted(merged["target_dataset"].unique().tolist()),
        "model_configs_covered": sorted(merged["model_config"].unique().tolist()),
    }
    return merged, audit_info


def compute_consolidated_tables(
    df_merged: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Compute and save:
      1. `consolidated_summary_by_dataset.csv`
      2. `consolidated_summary_global.csv`
      3. `consolidated_statistical_tests.csv` (within-dataset LOCO fold-level Wilcoxon + Holm)
      4. `consolidated_dataset_level_tests.csv` (dataset-level N=5 Friedman & Wilcoxon summary)
      5. `consolidated_rankings.csv` (per-dataset ranks and mean rank across N=5 datasets)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    def _agg_metrics(grp: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n_evaluations": int(len(grp)),
            "n_folds": int(grp["fold_id"].nunique()),
            "n_seeds": int(grp["seed"].nunique()),
        }
        for m_col in METRICS_COLS:
            s = grp[m_col].dropna()
            out[f"{m_col}_mean"] = float(s.mean()) if len(s) > 0 else float("nan")
            out[f"{m_col}_std"] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            out[f"{m_col}_median"] = float(s.median()) if len(s) > 0 else float("nan")
        return out

    # 1. consolidated_summary_by_dataset.csv
    ds_rows: List[Dict[str, Any]] = []
    for (ds, bal, arch, meth, m_cfg), grp in df_merged.groupby(
        ["target_dataset", "balance_strategy", "architecture", "method", "model_config"],
        sort=False,
    ):
        row = {
            "target_dataset": ds,
            "balance_strategy": bal,
            "architecture": arch,
            "method": meth,
            "model_config": m_cfg,
        }
        row.update(_agg_metrics(grp))
        ds_rows.append(row)
    df_sum_ds = pd.DataFrame(ds_rows)
    p_sum_ds = output_dir / "consolidated_summary_by_dataset.csv"
    df_sum_ds.to_csv(p_sum_ds, index=False)
    paths["summary_by_dataset"] = p_sum_ds

    # 2. consolidated_summary_global.csv
    # Computed both across all matched evaluations AND as the unweighted mean of dataset-level means (N=5 datasets)
    glob_rows: List[Dict[str, Any]] = []
    for (bal, arch, meth, m_cfg), grp in df_merged.groupby(
        ["balance_strategy", "architecture", "method", "model_config"], sort=False
    ):
        ds_sub = df_sum_ds[
            (df_sum_ds["balance_strategy"] == bal)
            & (df_sum_ds["model_config"] == m_cfg)
        ]
        row = {
            "balance_strategy": bal,
            "architecture": arch,
            "method": meth,
            "model_config": m_cfg,
            "n_datasets": int(grp["target_dataset"].nunique()),
            "total_evaluations": int(len(grp)),
            "dataset_macro_f1_mean_of_means": float(ds_sub["macro_f1_mean"].mean()),
            "dataset_macro_f1_std_across_datasets": float(ds_sub["macro_f1_mean"].std(ddof=1))
            if len(ds_sub) > 1
            else 0.0,
            "dataset_balanced_accuracy_mean_of_means": float(
                ds_sub["balanced_accuracy_mean"].mean()
            ),
            "dataset_roc_auc_mean_of_means": float(ds_sub["roc_auc_mean"].mean()),
        }
        row.update(_agg_metrics(grp))
        glob_rows.append(row)
    df_sum_glob = pd.DataFrame(glob_rows)
    p_sum_glob = output_dir / "consolidated_summary_global.csv"
    df_sum_glob.to_csv(p_sum_glob, index=False)
    paths["summary_global"] = p_sum_glob

    # 3. Within-dataset paired statistical tests at the seed-averaged LOCO fold level
    fold_level = (
        df_merged.groupby(
            [
                "target_dataset",
                "balance_strategy",
                "fold_id",
                "condition",
                "architecture",
                "method",
                "model_config",
            ],
            as_index=False,
        )[METRICS_COLS]
        .mean()
    )

    neural_focus = [
        "MLP (Scratch)",
        "MLP (Fine-Tuning)",
        "MLP (Linear Probing)",
        "TabNet (Scratch)",
        "TabNet (Fine-Tuning)",
        "TabNet (Linear Probing)",
    ]
    classical_focus = [
        "RandomForest (ClassicalBaseline)",
        "SVM (ClassicalBaseline)",
        "XGBoost (ClassicalBaseline)",
    ]

    comparison_pairs: List[Tuple[str, str, str]] = []
    # (a) Neural vs. Classical Baselines
    for n_cfg in neural_focus:
        for c_cfg in classical_focus:
            comparison_pairs.append(("neural_vs_classical", n_cfg, c_cfg))
    # (b) Classical vs. Classical
    comparison_pairs.extend(
        [
            ("classical_vs_classical", "XGBoost (ClassicalBaseline)", "RandomForest (ClassicalBaseline)"),
            ("classical_vs_classical", "RandomForest (ClassicalBaseline)", "SVM (ClassicalBaseline)"),
            ("classical_vs_classical", "XGBoost (ClassicalBaseline)", "SVM (ClassicalBaseline)"),
        ]
    )
    # (c) Neural TL vs. Neural Scratch
    comparison_pairs.extend(
        [
            ("tl_vs_scratch", "MLP (Fine-Tuning)", "MLP (Scratch)"),
            ("tl_vs_scratch", "MLP (Linear Probing)", "MLP (Scratch)"),
            ("tl_vs_scratch", "TabNet (Fine-Tuning)", "TabNet (Scratch)"),
            ("tl_vs_scratch", "TabNet (Linear Probing)", "TabNet (Scratch)"),
        ]
    )

    stat_rows: List[Dict[str, Any]] = []
    for (ds, bal), grp in fold_level.groupby(
        ["target_dataset", "balance_strategy"], sort=False
    ):
        for comp_family, cfg_a, cfg_b in comparison_pairs:
            sub_a = grp[grp["model_config"] == cfg_a].set_index("fold_id")
            sub_b = grp[grp["model_config"] == cfg_b].set_index("fold_id")
            if sub_a.empty or sub_b.empty:
                continue
            common_folds = sorted(
                set(sub_a.index.tolist()).intersection(set(sub_b.index.tolist()))
            )
            if not common_folds:
                continue

            for m_col in METRICS_COLS:
                a_vals = sub_a.loc[common_folds, m_col].to_numpy(dtype=np.float64)
                b_vals = sub_b.loc[common_folds, m_col].to_numpy(dtype=np.float64)
                valid_mask = ~(np.isnan(a_vals) | np.isnan(b_vals))
                a_v = a_vals[valid_mask]
                b_v = b_vals[valid_mask]
                n_paired = int(len(a_v))
                if n_paired == 0:
                    continue

                diffs = a_v - b_v
                mean_diff = float(np.mean(diffs))
                median_diff = float(np.median(diffs))
                std_diff = float(np.std(diffs, ddof=1)) if n_paired > 1 else 0.0
                if n_paired > 1 and std_diff > 0.0:
                    se = std_diff / math.sqrt(n_paired)
                    t_crit = float(stats.t.ppf(0.975, df=n_paired - 1))
                    ci_low = mean_diff - t_crit * se
                    ci_high = mean_diff + t_crit * se
                else:
                    ci_low = mean_diff
                    ci_high = mean_diff

                w_stat = float("nan")
                p_raw = float("nan")
                test_note = "OK_wilcoxon_signed_rank"
                if np.allclose(diffs, 0.0):
                    w_stat = 0.0
                    p_raw = 1.0
                    test_note = "All paired differences are identically zero"
                else:
                    try:
                        res_w = stats.wilcoxon(
                            diffs, zero_method="zsplit", alternative="two-sided"
                        )
                        w_stat = float(res_w.statistic)
                        p_raw = float(res_w.pvalue)
                        if n_paired < 6:
                            test_note = (
                                f"Low sample size (N={n_paired} LOCO folds < 6 minimum "
                                "for alpha=0.05 two-sided Wilcoxon significance)"
                            )
                    except Exception as exc:
                        test_note = f"Wilcoxon error: {exc}"

                stat_rows.append(
                    {
                        "target_dataset": ds,
                        "balance_strategy": bal,
                        "comparison_family": comp_family,
                        "comparison": f"{cfg_a} vs. {cfg_b}",
                        "model_a": cfg_a,
                        "model_b": cfg_b,
                        "metric": m_col,
                        "statistical_unit": "loco_fold_seed_averaged",
                        "n_paired_folds": n_paired,
                        "model_a_mean": float(np.mean(a_v)),
                        "model_b_mean": float(np.mean(b_v)),
                        "mean_diff_a_minus_b": mean_diff,
                        "median_diff_a_minus_b": median_diff,
                        "std_diff": std_diff,
                        "ci95_low": float(ci_low),
                        "ci95_high": float(ci_high),
                        "n_folds_a_better": int(np.sum(diffs > 1e-12)),
                        "n_folds_b_better": int(np.sum(diffs < -1e-12)),
                        "n_folds_tied": int(np.sum(np.abs(diffs) <= 1e-12)),
                        "wilcoxon_statistic": w_stat,
                        "p_value_raw": p_raw,
                        "p_value_holm": float("nan"),
                        "significant_alpha_0_05": False,
                        "methodological_note": test_note,
                    }
                )

    df_stats = pd.DataFrame(stat_rows)
    if not df_stats.empty:
        # Apply Holm step-down correction within each (target_dataset, balance_strategy, comparison_family, metric)
        for (_, _, _, _), idxs in df_stats.groupby(
            ["target_dataset", "balance_strategy", "comparison_family", "metric"]
        ).groups.items():
            idx_list = list(idxs)
            raw_ps = [float(df_stats.loc[i, "p_value_raw"]) for i in idx_list]
            adj_ps = _holm_bonferroni_adjust(raw_ps)
            for i, adj_p in zip(idx_list, adj_ps):
                df_stats.loc[i, "p_value_holm"] = adj_p
                df_stats.loc[i, "significant_alpha_0_05"] = bool(
                    not math.isnan(adj_p) and adj_p < 0.05
                )

    p_stats = output_dir / "consolidated_statistical_tests.csv"
    df_stats.to_csv(p_stats, index=False)
    paths["statistical_tests"] = p_stats

    # 4. Dataset-level rankings & Friedman omnibus test across the target datasets (N = n_datasets)
    rank_rows: List[Dict[str, Any]] = []
    ds_level_test_rows: List[Dict[str, Any]] = []

    if not df_sum_ds.empty:
        for (ds, bal), grp in df_sum_ds.groupby(
            ["target_dataset", "balance_strategy"], sort=False
        ):
            grp_sorted = grp.copy()
            grp_sorted["rank_all_models"] = grp_sorted["macro_f1_mean"].rank(
                ascending=False, method="average"
            )
            for _, r in grp_sorted.iterrows():
                rank_rows.append(
                    {
                        "scope": "dataset_all_models",
                        "target_dataset": ds,
                        "balance_strategy": bal,
                        "architecture": r["architecture"],
                        "method": r["method"],
                        "model_config": r["model_config"],
                        "metric": "macro_f1_mean",
                        "metric_value": float(r["macro_f1_mean"]),
                        "rank": float(r["rank_all_models"]),
                    }
                )

        df_ds_ranks = pd.DataFrame(rank_rows)
        if not df_ds_ranks.empty:
            for (bal, arch, meth, m_cfg), grp in df_ds_ranks.groupby(
                ["balance_strategy", "architecture", "method", "model_config"],
                sort=False,
            ):
                rank_rows.append(
                    {
                        "scope": "mean_rank_across_datasets",
                        "target_dataset": "ALL_TARGETS",
                        "balance_strategy": bal,
                        "architecture": arch,
                        "method": meth,
                        "model_config": m_cfg,
                        "metric": "macro_f1_mean",
                        "metric_value": float(grp["metric_value"].mean()),
                        "rank": float(grp["rank"].mean()),
                    }
                )

        # Friedman test at the dataset level (N = number of target datasets, per balance_strategy)
        for bal, grp in df_sum_ds.groupby("balance_strategy", sort=False):
            for m_col in ("macro_f1_mean", "balanced_accuracy_mean", "roc_auc_mean"):
                piv = grp.pivot(
                    index="target_dataset", columns="model_config", values=m_col
                ).dropna(axis=0, how="any")
                n_ds = int(len(piv))
                n_models = int(piv.shape[1])
                f_stat = float("nan")
                f_pval = float("nan")
                note = (
                    f"Dataset-level omnibus Friedman test across N={n_ds} target datasets "
                    f"and K={n_models} models (seeds and folds aggregated within each dataset first)."
                )
                if n_ds >= 2 and n_models >= 3:
                    try:
                        arrays = [piv[col].to_numpy(dtype=np.float64) for col in piv.columns]
                        res_f = stats.friedmanchisquare(*arrays)
                        f_stat = float(res_f.statistic)
                        f_pval = float(res_f.pvalue)
                    except Exception as exc:
                        note = f"Friedman test not applicable: {exc}"
                ds_level_test_rows.append(
                    {
                        "balance_strategy": bal,
                        "metric": m_col.replace("_mean", ""),
                        "statistical_unit": "target_dataset",
                        "n_datasets": n_ds,
                        "n_models": n_models,
                        "friedman_statistic": f_stat,
                        "friedman_p_value": f_pval,
                        "significant_alpha_0_05": bool(
                            not math.isnan(f_pval) and f_pval < 0.05
                        ),
                        "methodological_note": note,
                    }
                )

    df_ranks = pd.DataFrame(rank_rows)
    p_ranks = output_dir / "consolidated_rankings.csv"
    df_ranks.to_csv(p_ranks, index=False)
    paths["rankings"] = p_ranks

    df_ds_tests = pd.DataFrame(ds_level_test_rows)
    p_ds_tests = output_dir / "consolidated_dataset_level_tests.csv"
    df_ds_tests.to_csv(p_ds_tests, index=False)
    paths["dataset_level_tests"] = p_ds_tests

    return paths


def generate_consolidated_report_md(
    neural_dir: Path,
    classical_dir: Path,
    audit_info: Mapping[str, Any],
    df_sum_ds: pd.DataFrame,
    df_sum_glob: pd.DataFrame,
    df_stats: pd.DataFrame,
    df_ds_tests: pd.DataFrame,
    df_ranks: pd.DataFrame,
) -> str:
    """Render `consolidated_report.md` distinguishing descriptive rankings from inferential tests."""
    ds_cols = [
        "target_dataset",
        "balance_strategy",
        "model_config",
        "n_folds",
        "n_seeds",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "roc_auc_mean",
    ]
    glob_cols = [
        "balance_strategy",
        "model_config",
        "n_datasets",
        "total_evaluations",
        "dataset_macro_f1_mean_of_means",
        "dataset_macro_f1_std_across_datasets",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "roc_auc_mean",
    ]
    stat_cols = [
        "target_dataset",
        "balance_strategy",
        "comparison",
        "metric",
        "n_paired_folds",
        "model_a_mean",
        "model_b_mean",
        "mean_diff_a_minus_b",
        "p_value_raw",
        "p_value_holm",
        "significant_alpha_0_05",
    ]

    ds_table = _df_to_table_str(
        df_sum_ds[[c for c in ds_cols if c in df_sum_ds.columns]]
    )
    glob_table = _df_to_table_str(
        df_sum_glob[[c for c in glob_cols if c in df_sum_glob.columns]]
    )
    f1_stats = (
        df_stats[df_stats["metric"] == "macro_f1"]
        if not df_stats.empty and "metric" in df_stats.columns
        else df_stats
    )
    stats_table = _df_to_table_str(
        f1_stats[[c for c in stat_cols if c in f1_stats.columns]]
        if not f1_stats.empty
        else pd.DataFrame()
    )
    ds_tests_table = _df_to_table_str(df_ds_tests)
    mean_ranks_sub = (
        df_ranks[df_ranks["scope"] == "mean_rank_across_datasets"]
        if not df_ranks.empty and "scope" in df_ranks.columns
        else df_ranks
    )
    ranks_table = _df_to_table_str(mean_ranks_sub)

    return f"""# Relatório Consolidado — Baselines Clássicos (RF, SVM, XGBoost) vs. Redes Neurais Scratch e Transfer Learning

- **Diretório Neural/TL (Original Intocado)**: `{neural_dir}`
- **Diretório Baselines Clássicos**: `{classical_dir}`
- **Blocos `(target_dataset, fold_id, condition, seed, balance_strategy)` Pareados**: `{audit_info.get('matched_fold_seed_balance_blocks')}`
- **Total de Avaliações Consolidadas**: `{audit_info.get('merged_total_rows')}` (`Neural_DL={audit_info.get('neural_total_ok_rows')}`, `Classical_ML={audit_info.get('classical_total_ok_rows')}`)
- **Paridade de `split_signature` e `preprocessing_signature` Verificada**: `{audit_info.get('signatures_verified_identical')}`

---

## 1. Metodologia Estatística e Prevenção de Pseudorreplicação

1. **Separação Estrita de Unidades Experimentais**:
   - **Seeds (`42, 123, 2026`)**, **Folds LOCO ($K_T$)** e **Estratégias de Balanceamento (`none`, `smote`)** nunca são tratados como datasets independentes.
   - **Inferência Intra-Dataset (`consolidated_statistical_tests.csv`)**: Para cada `(target_dataset, balance_strategy)` separadamente, calcula-se primeiro a média das 3 seeds dentro de cada fold LOCO (`loco_fold_seed_averaged`, $N = K_T$ folds), aplicando-se o teste pareado bicaudal de Wilcoxon signed-rank com correção *step-down* de Holm-Bonferroni (`p_value_holm`).
   - **Análise de Generalização Entre Datasets (`consolidated_rankings.csv` e `consolidated_dataset_level_tests.csv`)**: Mantida estritamente no nível do dataset ($N = 5$ datasets alvo: `CWRU_12k`, `CWRU_48k`, `HUST_Gearbox`, `UOEMD`, `PU`), avaliada separadamente para `none` e `smote` mediante rankings por dataset, média de ranks entre os 5 datasets e teste global de Friedman.

---

## 2. Resultados Consolidados por Dataset Alvo (`Dataset-Level Summary`)

```text
{ds_table}
```

---

## 3. Resumo Global e Média entre os 5 Datasets Alvo (`Global & Cross-Dataset Summary`)

```text
{glob_table}
```

---

## 4. Rankings Médios no Nível de Dataset ($N = 5$ Datasets Alvo, `Macro-F1`)

```text
{ranks_table}
```

---

## 5. Teste Omnibus de Friedman no Nível de Dataset ($N = 5$ Datasets)

```text
{ds_tests_table}
```

---

## 6. Testes Pareados Intra-Dataset (Wilcoxon Signed-Rank + Holm, `Macro-F1`)

> **Distinção entre Ranking Descritivo e Significância Estatística**: Nos datasets com $K_T = 4$ condições operacionais (`CWRU_12k` e `PU`), o menor $p$-valor bicaudal exato possível no teste de Wilcoxon é $2/2^4 = 0.125 > 0.05$; portanto, nesses dois datasets as diferenças devem ser interpretadas descritivamente (média, IC 95% e proporção de folds vencidos), enquanto `CWRU_48k` ($K_T = 12$), `UOEMD` ($K_T = 16$) e `HUST_Gearbox` ($K_T = 30$) possuem potência estatística formal em $\\alpha = 0.05$ após correção de Holm.

```text
{stats_table}
```
"""


def run_consolidated_comparison(
    neural_run_dir: str | Path = DEFAULT_REFERENCE_BENCHMARK_DIR,
    classical_run_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """
    Combine the completed Neural (`MLP`, `TabNet`) benchmark and Classical
    (`RandomForest`, `SVM`, `XGBoost`) benchmark into a new consolidated analysis
    directory without modifying either source directory.
    """
    n_dir = Path(neural_run_dir)
    if not (n_dir / "results.csv").exists():
        raise FileNotFoundError(f"Neural benchmark results.csv not found in '{n_dir}'.")

    c_dir = (
        Path(classical_run_dir)
        if classical_run_dir is not None
        else resolve_latest_classical_run_dir()
    )
    if not (c_dir / "results.csv").exists():
        raise FileNotFoundError(f"Classical baseline results.csv not found in '{c_dir}'.")

    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = DEFAULT_CONSOLIDATED_ROOT / f"run_{base_ts}"
        suffix = 1
        while out_dir.exists():
            out_dir = DEFAULT_CONSOLIDATED_ROOT / f"run_{base_ts}_{suffix:02d}"
            suffix += 1
        out_dir.mkdir(parents=True, exist_ok=False)

    df_neural = pd.read_csv(n_dir / "results.csv")
    df_classical = pd.read_csv(c_dir / "results.csv")

    df_merged, audit_info = verify_and_merge_matched_results(df_neural, df_classical)
    df_merged.to_csv(out_dir / "consolidated_results.csv", index=False)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "neural_run_dir": str(n_dir),
        "classical_run_dir": str(c_dir),
        "output_dir": str(out_dir),
        "audit": audit_info,
    }
    (out_dir / "consolidation_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    paths = compute_consolidated_tables(df_merged, out_dir)
    df_sum_ds = pd.read_csv(paths["summary_by_dataset"])
    df_sum_glob = pd.read_csv(paths["summary_global"])
    df_stats = (
        pd.read_csv(paths["statistical_tests"])
        if paths["statistical_tests"].exists()
        and paths["statistical_tests"].stat().st_size > 1
        else pd.DataFrame()
    )
    df_ds_tests = (
        pd.read_csv(paths["dataset_level_tests"])
        if paths["dataset_level_tests"].exists()
        and paths["dataset_level_tests"].stat().st_size > 1
        else pd.DataFrame()
    )
    df_ranks = (
        pd.read_csv(paths["rankings"])
        if paths["rankings"].exists() and paths["rankings"].stat().st_size > 1
        else pd.DataFrame()
    )

    report_md = generate_consolidated_report_md(
        neural_dir=n_dir,
        classical_dir=c_dir,
        audit_info=audit_info,
        df_sum_ds=df_sum_ds,
        df_sum_glob=df_sum_glob,
        df_stats=df_stats,
        df_ds_tests=df_ds_tests,
        df_ranks=df_ranks,
    )
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    return out_dir


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate and statistically compare MLP/TabNet (Scratch, Fine-Tuning, "
            "Linear Probing) with Classical Baselines (RandomForest, SVM, XGBoost)."
        )
    )
    parser.add_argument(
        "--neural-run-dir",
        type=Path,
        default=DEFAULT_REFERENCE_BENCHMARK_DIR,
        help="Path to the completed neural benchmark directory (default: results/scratch_vs_tl/run_20261105_141743).",
    )
    parser.add_argument(
        "--classical-run-dir",
        type=Path,
        default=None,
        help="Path to the completed classical baselines run_<timestamp> directory (default: latest under results/scratch_vs_tl/classical_baselines/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for the consolidated comparison.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = run_consolidated_comparison(
        neural_run_dir=args.neural_run_dir,
        classical_run_dir=args.classical_run_dir,
        output_dir=args.output_dir,
    )
    print(f"Consolidated comparison generated at: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

