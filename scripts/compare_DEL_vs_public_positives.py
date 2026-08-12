#!/usr/bin/env python3
"""Compare DEL positives with public positives used for control fine-tuning."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

BASE = Path("/share/home/u25511/wangyan")
OUT_DIR = BASE / "all9_comprehensive_finetune_analysis" / "public_data_control" / "DEL_vs_public_positive_value_analysis"

TARGETS = {
    "BRD4": {
        "del_dir": BASE / "HitScreen" / "data" / "finetune_DEL_samples_filtered" / "BDR4",
        "public_dir": BASE / "HitScreen" / "data" / "finetune_public_controls" / "BRD4_PublicPos_PubChem_DUDEall_1to10",
        "test_csv": BASE / "CHEMBL_with_DUDE_csv_20260715" / "BRD4_CHEMBL10_with_DUDE.csv",
    },
    "MAPK14": {
        "del_dir": BASE / "HitScreen" / "data" / "finetune_DEL_samples_filtered" / "MAPK14",
        "public_dir": BASE / "HitScreen" / "data" / "finetune_public_controls" / "MAPK14_PublicPos_PubChem_DUDEall_1to10",
        "test_csv": BASE / "CHEMBL_with_DUDE_csv_20260715" / "MAPK14_CHEMBL_with_DUDE.csv",
    },
    "CAIX": {
        "del_dir": BASE / "HitScreen" / "data" / "finetune_DEL_samples_filtered" / "CAIX",
        "public_dir": BASE / "HitScreen" / "data" / "finetune_public_controls" / "CAIX_PublicPos_BindingDB_PubChem_DUDEall_1to10",
        "test_csv": BASE / "CHEMBL_with_DUDE_csv_20260715" / "CAIX_CHEMBL_active_with_DUDE_1to10.csv",
    },
}


def smiles_col(df: pd.DataFrame) -> str:
    for col in ["SMILES", "smiles", "canonical_smiles", "Canonical_SMILES", "compound_iso_smiles"]:
        if col in df.columns:
            return col
    for col in df.columns:
        if "smiles" in col.lower():
            return col
    raise ValueError(f"No SMILES column in columns: {df.columns.tolist()}")


def label_col(df: pd.DataFrame) -> str | None:
    for col in ["Y", "y", "label", "Label", "activity"]:
        if col in df.columns:
            return col
    return None


def canon(smi: str) -> str | None:
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def scaffold(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaf, canonical=True)


def fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def props(smi: str) -> dict | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
    }


def load_positive_from_trainval(data_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "val.csv")]
    df = pd.concat(frames, ignore_index=True)
    sc = smiles_col(df)
    lc = label_col(df)
    if lc is not None:
        df = df[df[lc].eq(1)].copy()
    out = pd.DataFrame({"SMILES": df[sc].dropna().astype(str)})
    out["canonical_smiles"] = out["SMILES"].map(canon)
    return out.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles").reset_index(drop=True)


def load_test_active(test_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(test_csv)
    sc = smiles_col(df)
    lc = label_col(df)
    if lc is None:
        raise ValueError(f"{test_csv} lacks label column")
    active = df[df[lc].eq(1)].copy()
    out = pd.DataFrame({"SMILES": active[sc].dropna().astype(str)})
    out["canonical_smiles"] = out["SMILES"].map(canon)
    return out.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles").reset_index(drop=True)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].map(scaffold)
    prop_rows = [props(smi) for smi in df["canonical_smiles"]]
    prop_df = pd.DataFrame(prop_rows)
    return pd.concat([df, prop_df], axis=1)


def nearest_similarity(query: pd.DataFrame, ref: pd.DataFrame, target: str, ref_name: str) -> pd.DataFrame:
    ref_fps = [fp(s) for s in ref["canonical_smiles"]]
    ref_fps = [x for x in ref_fps if x is not None]
    values = []
    nearest_smiles = []
    for smi in query["canonical_smiles"]:
        qfp = fp(smi)
        if qfp is None or not ref_fps:
            values.append(np.nan)
            nearest_smiles.append("")
            continue
        sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)
        idx = int(np.argmax(sims))
        values.append(float(sims[idx]))
        nearest_smiles.append(ref["canonical_smiles"].iloc[idx])
    return pd.DataFrame(
        {
            "target": target,
            "ref_set": ref_name,
            "test_active_smiles": query["canonical_smiles"],
            "nearest_ref_smiles": nearest_smiles,
            "nearest_tanimoto": values,
        }
    )


def summarize_set(target: str, set_name: str, df: pd.DataFrame, test_active: pd.DataFrame) -> dict:
    scaf_counts = df["scaffold"].value_counts(dropna=False)
    test_scaf = set(test_active["scaffold"])
    own_scaf = set(df["scaffold"])
    shared_scaf = own_scaf & test_scaf
    test_covered = test_active["scaffold"].isin(shared_scaf).sum()
    return {
        "target": target,
        "positive_set": set_name,
        "n_unique_molecules": int(len(df)),
        "n_unique_scaffolds": int(df["scaffold"].nunique(dropna=False)),
        "scaffold_ratio": float(df["scaffold"].nunique(dropna=False) / max(len(df), 1)),
        "top1_scaffold_fraction": float(scaf_counts.iloc[0] / max(len(df), 1)) if len(scaf_counts) else np.nan,
        "top5_scaffold_fraction": float(scaf_counts.iloc[:5].sum() / max(len(df), 1)) if len(scaf_counts) else np.nan,
        "shared_scaffolds_with_test_active": int(len(shared_scaf)),
        "test_active_scaffold_coverage_fraction": float(test_covered / max(len(test_active), 1)),
        "MW_median": float(df["MW"].median()),
        "LogP_median": float(df["LogP"].median()),
        "TPSA_median": float(df["TPSA"].median()),
        "RotB_median": float(df["RotB"].median()),
    }


def summarize_similarity(sim: pd.DataFrame) -> dict:
    x = sim["nearest_tanimoto"].dropna()
    return {
        "target": sim["target"].iloc[0],
        "positive_set": sim["ref_set"].iloc[0],
        "test_active_nearest_similarity_median": float(x.median()),
        "test_active_nearest_similarity_q75": float(x.quantile(0.75)),
        "test_active_nearest_similarity_q90": float(x.quantile(0.90)),
        "test_active_coverage_sim_ge_0.3": float((x >= 0.3).mean()),
        "test_active_coverage_sim_ge_0.4": float((x >= 0.4).mean()),
        "test_active_coverage_sim_ge_0.5": float((x >= 0.5).mean()),
        "test_active_coverage_sim_ge_0.6": float((x >= 0.6).mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_rows = []
    sim_rows = []
    all_sim = []
    overlap_rows = []
    prop_rows = []

    for target, cfg in TARGETS.items():
        del_pos = annotate(load_positive_from_trainval(cfg["del_dir"]))
        public_pos = annotate(load_positive_from_trainval(cfg["public_dir"]))
        test_active = annotate(load_test_active(cfg["test_csv"]))

        del_pos.to_csv(OUT_DIR / f"{target}_DEL_trainval_positive_annotated.csv", index=False)
        public_pos.to_csv(OUT_DIR / f"{target}_public_trainval_positive_annotated.csv", index=False)
        test_active.to_csv(OUT_DIR / f"{target}_test_active_annotated.csv", index=False)

        set_rows.append(summarize_set(target, "DEL_positive", del_pos, test_active))
        set_rows.append(summarize_set(target, "Public_positive", public_pos, test_active))

        for name, ref in [("DEL_positive", del_pos), ("Public_positive", public_pos)]:
            sim = nearest_similarity(test_active, ref, target, name)
            sim.to_csv(OUT_DIR / f"{target}_test_active_nearest_{name}.csv", index=False)
            all_sim.append(sim)
            sim_rows.append(summarize_similarity(sim))

        del_can = set(del_pos["canonical_smiles"])
        pub_can = set(public_pos["canonical_smiles"])
        del_scaf = set(del_pos["scaffold"])
        pub_scaf = set(public_pos["scaffold"])
        overlap_rows.append(
            {
                "target": target,
                "molecule_overlap_DEL_public": int(len(del_can & pub_can)),
                "molecule_overlap_fraction_of_DEL": float(len(del_can & pub_can) / max(len(del_can), 1)),
                "molecule_overlap_fraction_of_public": float(len(del_can & pub_can) / max(len(pub_can), 1)),
                "scaffold_overlap_DEL_public": int(len(del_scaf & pub_scaf)),
                "scaffold_overlap_fraction_of_DEL": float(len(del_scaf & pub_scaf) / max(len(del_scaf), 1)),
                "scaffold_overlap_fraction_of_public": float(len(del_scaf & pub_scaf) / max(len(pub_scaf), 1)),
            }
        )

        for set_name, df in [("DEL_positive", del_pos), ("Public_positive", public_pos), ("Test_active", test_active)]:
            tmp = df[["MW", "LogP", "TPSA", "HBA", "HBD", "RotB", "HeavyAtomCount"]].copy()
            tmp["target"] = target
            tmp["set"] = set_name
            prop_rows.append(tmp)

    set_summary = pd.DataFrame(set_rows)
    sim_summary = pd.DataFrame(sim_rows)
    overlap_summary = pd.DataFrame(overlap_rows)
    merged = set_summary.merge(sim_summary, on=["target", "positive_set"], how="left")

    set_summary.to_csv(OUT_DIR / "DEL_vs_public_positive_set_summary.csv", index=False)
    sim_summary.to_csv(OUT_DIR / "DEL_vs_public_test_active_similarity_summary.csv", index=False)
    overlap_summary.to_csv(OUT_DIR / "DEL_vs_public_positive_overlap_summary.csv", index=False)
    merged.to_csv(OUT_DIR / "DEL_vs_public_positive_value_summary.csv", index=False)
    all_sim_df = pd.concat(all_sim, ignore_index=True)
    all_sim_df.to_csv(OUT_DIR / "DEL_vs_public_test_active_nearest_similarity_all.csv", index=False)
    prop_df = pd.concat(prop_rows, ignore_index=True)
    prop_df.to_csv(OUT_DIR / "DEL_vs_public_test_physchem_values.csv", index=False)

    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.boxplot(data=all_sim_df, x="target", y="nearest_tanimoto", hue="ref_set", ax=ax, showfliers=False)
    ax.set_xlabel("")
    ax.set_ylabel("Nearest Tanimoto from test active")
    ax.set_title("Test Active Coverage by DEL vs Public Positives")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig1_test_active_nearest_similarity_DEL_vs_public.png", dpi=300)
    fig.savefig(OUT_DIR / "Fig1_test_active_nearest_similarity_DEL_vs_public.pdf")

    cov = merged.melt(
        id_vars=["target", "positive_set"],
        value_vars=[
            "test_active_coverage_sim_ge_0.3",
            "test_active_coverage_sim_ge_0.4",
            "test_active_coverage_sim_ge_0.5",
        ],
        var_name="coverage_threshold",
        value_name="fraction",
    )
    cov["coverage_threshold"] = cov["coverage_threshold"].str.replace("test_active_coverage_sim_ge_", "sim >= ", regex=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    sns.barplot(data=cov, x="target", y="fraction", hue="positive_set", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of test actives covered")
    ax.set_title("Coverage of Test Actives at Similarity Thresholds")
    ax.legend(title="")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "Fig2_test_active_similarity_threshold_coverage.png", dpi=300)
    fig.savefig(OUT_DIR / "Fig2_test_active_similarity_threshold_coverage.pdf")

    scaf = merged.melt(
        id_vars=["target", "positive_set"],
        value_vars=["shared_scaffolds_with_test_active", "test_active_scaffold_coverage_fraction"],
        var_name="metric",
        value_name="value",
    )
    g = sns.catplot(data=scaf, x="target", y="value", hue="positive_set", col="metric", kind="bar", sharey=False, height=4, aspect=1.1)
    g.set_axis_labels("", "")
    g.set_titles("{col_name}")
    g.savefig(OUT_DIR / "Fig3_test_active_scaffold_coverage_DEL_vs_public.png", dpi=300)
    g.savefig(OUT_DIR / "Fig3_test_active_scaffold_coverage_DEL_vs_public.pdf")
    plt.close("all")

    prop_medians = prop_df.groupby(["target", "set"])[["MW", "LogP", "TPSA", "RotB"]].median().reset_index()
    prop_medians.to_csv(OUT_DIR / "DEL_vs_public_test_physchem_medians.csv", index=False)
    print(f"Saved outputs to {OUT_DIR}")
    print(merged.round(4).to_string(index=False))
    print(overlap_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
