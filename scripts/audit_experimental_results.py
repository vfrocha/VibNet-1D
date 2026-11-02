#!/usr/bin/env python3
"""Inventory result artifacts and summarize recognized metrics without merging incompatible protocols."""
import argparse
import re
from pathlib import Path
import pandas as pd

ALIASES = {
    "macro_f1": ["macro_f1", "macro f1", "f1_macro", "macro-f1"],
    "accuracy": ["accuracy", "acc", "accuracy_score"],
    "balanced_accuracy": ["balanced_accuracy", "bal_acc", "balanced acc", "bal acc"],
    "roc_auc": ["roc_auc", "roc-auc", "auc"],
}
DATASET = ["dataset", "target", "target_dataset", "test_dataset"]
MODEL = ["model", "classifier", "estimator", "architecture"]
SOURCE = ["source", "sources", "source_dataset", "source_datasets"]
STRATEGY = ["strategy", "variant", "preprocessing", "preprocess", "balance", "balancing"]

def norm(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")

def col(columns, aliases):
    mapping = {norm(c): c for c in columns}
    for a in aliases:
        if norm(a) in mapping:
            return mapping[norm(a)]
    return None

def read_csv(path):
    err = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, sep=None, engine="python", on_bad_lines="warn")
        except Exception as e:
            err = e
    raise err

def family(path, root):
    parts = path.relative_to(root).parts
    for p in parts[:-1]:
        if p.lower() in {"baseline", "tl_cross_dataset_mlp_tabnet", "tl_contrastive_mlp", "tabvibnet", "time_domain"}:
            return p
    return parts[0] if len(parts) > 1 else "root"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--output-dir", type=Path, default=Path("reports/reproducibility_audit"))
    args = ap.parse_args()
    root, out = args.results_dir.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Results directory not found: {root}")
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md", ".txt", ".yaml", ".yml"})
    inventory, metric_rows = [], []
    for p in files:
        rec = {"relative_path": str(p.relative_to(root)), "family": family(p, root),
               "type": p.suffix.lower(), "size_bytes": p.stat().st_size,
               "rows": None, "columns": None, "status": "inventory_only", "notes": ""}
        if p.suffix.lower() == ".csv":
            try:
                df = read_csv(p)
                rec.update(rows=len(df), columns=len(df.columns), status="ok")
                dcol, mcol = col(df.columns, DATASET), col(df.columns, MODEL)
                rec["dataset_column"], rec["model_column"] = dcol, mcol
                for metric, aliases in ALIASES.items():
                    mc = col(df.columns, aliases)
                    if mc is None:
                        continue
                    for i, row in df.iterrows():
                        value = pd.to_numeric(pd.Series([row[mc]]), errors="coerce").iloc[0]
                        if pd.isna(value):
                            continue
                        sc, st = col(df.columns, SOURCE), col(df.columns, STRATEGY)
                        metric_rows.append({
                            "relative_path": rec["relative_path"], "family": rec["family"],
                            "row_index": int(i), "dataset": str(row[dcol]) if dcol and pd.notna(row[dcol]) else "",
                            "model": str(row[mcol]) if mcol and pd.notna(row[mcol]) else "",
                            "sources": str(row[sc]) if sc and pd.notna(row[sc]) else "",
                            "strategy": str(row[st]) if st and pd.notna(row[st]) else "",
                            "metric": metric, "value": float(value)})
                if not any(col(df.columns, a) for a in ALIASES.values()):
                    rec["notes"] = "No recognized metric columns; inventory only."
            except Exception as e:
                rec.update(status="error", notes=str(e)[:400])
        inventory.append(rec)
    inv = pd.DataFrame(inventory)
    long = pd.DataFrame(metric_rows, columns=["relative_path","family","row_index","dataset","model","sources","strategy","metric","value"])
    inv.to_csv(out / "results_inventory.csv", index=False)
    long.to_csv(out / "metrics_long.csv", index=False)
    if long.empty:
        summary = pd.DataFrame(columns=["family","relative_path","dataset","model","sources","strategy","metric","count","mean","std","min","max"])
    else:
        summary = long.groupby(["family","relative_path","dataset","model","sources","strategy","metric"], dropna=False)["value"].agg(["count","mean","std","min","max"]).reset_index()
    summary.to_csv(out / "metrics_summary.csv", index=False)
    errors = inv[inv.status == "error"] if not inv.empty else pd.DataFrame()
    report = [
        "# VibNet-1D — Auditoria de resultados", "",
        f"- Diretório: `{root}`", f"- Arquivos inventariados: {len(files)}",
        f"- CSVs: {sum(p.suffix.lower()=='.csv' for p in files)}",
        f"- Linhas de métricas extraídas: {len(long)}", f"- Erros de leitura: {len(errors)}", "",
        "## Arquivos gerados",
        "- `results_inventory.csv`: inventário e status de leitura.",
        "- `metrics_long.csv`: métricas reconhecidas com caminho de origem.",
        "- `metrics_summary.csv`: estatísticas descritivas por arquivo/dataset/modelo/estratégia.", "",
        "## Limitações e cautelas",
        "- Não compare resultados de arquivos diferentes sem verificar folds, seeds, dataset, pré-processamento, balanceamento e protocolo.",
        "- Linhas por fold e linhas agregadas podem coexistir; verifique a granularidade antes de resumir.",
        "- Métricas/colunas com nomes desconhecidos podem não ser extraídas automaticamente.",
        "- Resultados originais não são alterados; relatórios são derivados.", "",
        "## Erros de leitura"
    ]
    report += [f"- `{r.relative_path}`: {r.notes}" for _, r in errors.iterrows()] if len(errors) else ["Nenhum."]
    (out / "EXPERIMENTAL_RESULTS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Audit complete: {out} | files={len(files)} | metric rows={len(long)} | errors={len(errors)}")

if __name__ == "__main__":
    main()

