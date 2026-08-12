#!/usr/bin/env python3
"""Compare test-set active molecules captured in top 1% after DEL/public fine-tuning."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

BASE = Path("/share/home/u25511/wangyan")
OUT_DIR = (
    BASE
    / "all9_comprehensive_finetune_analysis"
    / "public_data_control"
    / "DEL_vs_public_top1_active_hit_analysis"
)

PUBLIC_TOP1 = {
    "BRD4": BASE
    / "GraphBAN/result/BRD4_MAPK14_PublicControls_DUDEall_1to10_graphban_coldv4_ft_lr5e6_ep10/BRD4"
    / "BRD4_PublicControls_DUDEall_1to10_v4pretrain_freeze_bcnmlp_lr5e6_ep10/screen"
    / "BRD4_CHEMBL10_BRD4_PublicControls_DUDEall_1to10_v4pretrain_freeze_bcnmlp_lr5e6_ep10_top1_percent_hits.csv",
    "MAPK14": BASE
    / "GraphBAN/result/BRD4_MAPK14_PublicControls_DUDEall_1to10_graphban_coldv4_ft_lr5e6_ep10/MAPK14"
    / "MAPK14_PublicControls_DUDEall_1to10_v4pretrain_freeze_bcnmlp_lr5e6_ep10/screen"
    / "MAPK14_CHEMBL10_MAPK14_PublicControls_DUDEall_1to10_v4pretrain_freeze_bcnmlp_lr5e6_ep10_top1_percent_hits.csv",
    "CAIX": BASE
    / "GraphBAN/result/CAIX_PublicControls_DUDEall_1to10_graphban_coldv4_ft_lr5e6_ep10"
    / "CAIX_PublicControls_DUDEall_1to10_v4pretrain_freeze_bcnmlp_lr5e6_ep10/screen"
    / "CAIX_CHEMBL10_CAIX_PublicControls_DUDEall_1to10_v4pretrain_freeze_bcnmlp_lr5e6_ep10_top1_percent_hits.csv",
}


def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaf, canonical=True)


def properties(smi: str) -> dict | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
    }


def top1_from_summary(summary_path: str | Path) -> Path:
    return Path(str(summary_path).replace("_summary.csv", "_top1_percent_hits.csv"))


def load_del_top1_paths() -> dict[str, Path]:
    metrics = pd.read_csv(BASE / "all9_comprehensive_finetune_analysis/all9_7model_comprehensive_metrics.csv")
    paths = {}
    for target in ["BRD4", "MAPK14", "CAIX"]:
        row = metrics[
            metrics["protein"].eq(target)
            & metrics["model"].eq("GraphBAN")
            & metrics["stage"].eq("Best fine-tune")
        ].iloc[0]
        paths[target] = top1_from_summary(row["source_file"])
    return paths


def load_top1_active(path: Path, target: str, setting: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Y" not in df.columns:
        raise ValueError(f"{path} lacks Y column")
    active = df[df["Y"].eq(1)].copy()
    active = active[["SMILES", "Y", "pred"]].copy()
    active["canonical_smiles"] = active["SMILES"].map(canonical_smiles)
    active = active.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles").reset_index(drop=True)
    active["scaffold"] = active["canonical_smiles"].map(scaffold)
    prop_df = pd.DataFrame([properties(s) for s in active["canonical_smiles"]])
    active = pd.concat([active, prop_df], axis=1)
    active.insert(0, "target", target)
    active.insert(1, "setting", setting)
    return active


def summarize_hits(df: pd.DataFrame, total_top1: int, target: str, setting: str) -> dict:
    scaf_counts = df["scaffold"].value_counts(dropna=False)
    return {
        "target": target,
        "setting": setting,
        "top1_total_molecules": int(total_top1),
        "active_hits": int(len(df)),
        "top1_precision": float(len(df) / max(total_top1, 1)),
        "unique_active_scaffolds": int(df["scaffold"].nunique(dropna=False)),
        "scaffold_per_active_hit": float(df["scaffold"].nunique(dropna=False) / max(len(df), 1)),
        "top1_scaffold_fraction_among_hits": float(scaf_counts.iloc[0] / max(len(df), 1)) if len(scaf_counts) else np.nan,
        "top5_scaffold_fraction_among_hits": float(scaf_counts.iloc[:5].sum() / max(len(df), 1)) if len(scaf_counts) else np.nan,
        "median_pred_among_active_hits": float(df["pred"].median()) if len(df) else np.nan,
        "MW_median": float(df["MW"].median()) if len(df) else np.nan,
        "LogP_median": float(df["LogP"].median()) if len(df) else np.nan,
        "TPSA_median": float(df["TPSA"].median()) if len(df) else np.nan,
        "RotB_median": float(df["RotB"].median()) if len(df) else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    del_paths = load_del_top1_paths()
    all_hits = []
    summary_rows = []
    overlap_rows = []

    for target in ["BRD4", "MAPK14", "CAIX"]:
        target_hits = {}
        for setting, path in [("DEL_best_finetune", del_paths[target]), ("Public_finetune", PUBLIC_TOP1[target])]:
            raw = pd.read_csv(path)
            hits = load_top1_active(path, target, setting)
            hits.to_csv(OUT_DIR / f"{target}_{setting}_top1_active_hits.csv", index=False)
            target_hits[setting] = hits
            all_hits.append(hits)
            summary_rows.append(summarize_hits(hits, len(raw), target, setting))

        del_hits = target_hits["DEL_best_finetune"]
        pub_hits = target_hits["Public_finetune"]
        del_mols = set(del_hits["canonical_smiles"])
        pub_mols = set(pub_hits["canonical_smiles"])
        del_scaf = set(del_hits["scaffold"])
        pub_scaf = set(pub_hits["scaffold"])
        overlap_rows.append(
            {
                "target": target,
                "DEL_active_hits": int(len(del_mols)),
                "Public_active_hits": int(len(pub_mols)),
                "shared_active_hits": int(len(del_mols & pub_mols)),
                "DEL_only_active_hits": int(len(del_mols - pub_mols)),
                "Public_only_active_hits": int(len(pub_mols - del_mols)),
                "shared_hit_fraction_of_DEL": float(len(del_mols & pub_mols) / max(len(del_mols), 1)),
                "shared_hit_fraction_of_Public": float(len(del_mols & pub_mols) / max(len(pub_mols), 1)),
                "DEL_active_scaffolds": int(len(del_scaf)),
                "Public_active_scaffolds": int(len(pub_scaf)),
                "shared_active_scaffolds": int(len(del_scaf & pub_scaf)),
                "DEL_only_active_scaffolds": int(len(del_scaf - pub_scaf)),
                "Public_only_active_scaffolds": int(len(pub_scaf - del_scaf)),
            }
        )

    hits_all = pd.concat(all_hits, ignore_index=True)
    hit_summary = pd.DataFrame(summary_rows)
    overlap_summary = pd.DataFrame(overlap_rows)

    hits_all.to_csv(OUT_DIR / "DEL_vs_public_top1_active_hits_all.csv", index=False)
    hit_summary.to_csv(OUT_DIR / "DEL_vs_public_top1_active_hit_summary.csv", index=False)
    overlap_summary.to_csv(OUT_DIR / "DEL_vs_public_top1_active_hit_overlap_summary.csv", index=False)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = {"DEL_best_finetune": "#4c78a8", "Public_finetune": "#f58518"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, metric, title in [
        (axes[0], "active_hits", "Top 1% Active Hits"),
        (axes[1], "unique_active_scaffolds", "Active-Hit Scaffolds"),
        (axes[2], "top1_precision", "Top 1% Precision"),
    ]:
        sns.barplot(data=hit_summary, x="target", y=metric, hue="setting", palette=palette, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.legend_.remove()
        for container in ax.containers:
            fmt = "%.2f" if metric == "top1_precision" else "%.0f"
            ax.bar_label(container, fmt=fmt, fontsize=8, padding=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.savefig(OUT_DIR / "Fig1_DEL_vs_public_top1_active_hit_counts.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "Fig1_DEL_vs_public_top1_active_hit_counts.pdf", bbox_inches="tight")

    overlap_plot = overlap_summary.melt(
        id_vars="target",
        value_vars=["shared_active_hits", "DEL_only_active_hits", "Public_only_active_hits"],
        var_name="hit_group",
        value_name="n_hits",
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(data=overlap_plot, x="target", y="n_hits", hue="hit_group", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Number of active hits")
    ax.set_title("Overlap of Top 1% Active Hits")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_DEL_vs_public_top1_active_hit_overlap.png", dpi=300)
    fig.savefig(OUT_DIR / "Fig2_DEL_vs_public_top1_active_hit_overlap.pdf")

    prop = hits_all.melt(
        id_vars=["target", "setting"],
        value_vars=["MW", "LogP", "TPSA", "RotB"],
        var_name="property",
        value_name="value",
    )
    g = sns.catplot(
        data=prop,
        x="target",
        y="value",
        hue="setting",
        col="property",
        kind="box",
        showfliers=False,
        sharey=False,
        height=3.6,
        aspect=0.9,
        palette=palette,
    )
    g.set_axis_labels("", "")
    g.set_titles("{col_name}")
    g.savefig(OUT_DIR / "Fig3_DEL_vs_public_top1_active_hit_physchem.png", dpi=300)
    g.savefig(OUT_DIR / "Fig3_DEL_vs_public_top1_active_hit_physchem.pdf")
    plt.close("all")

    print(f"Saved outputs to {OUT_DIR}")
    print(hit_summary.round(4).to_string(index=False))
    print(overlap_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
