#!/usr/bin/env python3
"""Select a global MW-matched DeepCoy decoy set for a ChEMBL test set."""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors


RDLogger.DisableLog("rdApp.*")
OUTPUT_COLUMNS = ["ID", "SMILES", "Y", "Protein", "target_cluster"]


def canonical_smiles(smi):
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


def mol_props(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "RB": Descriptors.NumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
    }


def fingerprint(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def read_generated_pairs(path):
    rows = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            toks = line.strip().split()
            if len(toks) >= 2:
                rows.append({"source_active": toks[0], "decoy_smiles": toks[1]})
    if not rows:
        raise RuntimeError("No generated pairs found in %s" % path)
    return pd.DataFrame(rows)


def load_excluded(paths):
    excluded = set()
    for path in paths:
        if path is None:
            continue
        df = pd.read_csv(path)
        smi_col = "SMILES" if "SMILES" in df.columns else "Ligand" if "Ligand" in df.columns else None
        if smi_col:
            excluded |= set(df[smi_col].dropna().map(canonical_smiles).dropna())
    return excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--generated-pairs", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max-source-tanimoto", type=float, default=0.3)
    parser.add_argument("--mw-bin-width", type=float, default=25.0)
    parser.add_argument("--exclude-csv", action="append", default=[])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(args.test_csv)
    missing = set(OUTPUT_COLUMNS) - set(test.columns)
    if missing:
        raise ValueError("%s lacks columns: %s" % (args.test_csv, missing))

    actives = test.loc[test["Y"].eq(1), OUTPUT_COLUMNS].copy().reset_index(drop=True)
    actives["can"] = actives["SMILES"].map(canonical_smiles)
    actives["props"] = actives["can"].map(mol_props)
    actives = actives.dropna(subset=["can", "props"]).drop_duplicates("can").reset_index(drop=True)
    actives["MW"] = actives["props"].map(lambda x: x["MW"])
    target_n = len(actives) * args.ratio

    excluded = load_excluded([Path(p) for p in args.exclude_csv])
    excluded |= set(actives["can"])

    generated = read_generated_pairs(args.generated_pairs)
    generated["source_can"] = generated["source_active"].map(canonical_smiles)
    generated["decoy_can"] = generated["decoy_smiles"].map(canonical_smiles)
    generated = generated.dropna(subset=["source_can", "decoy_can"])
    generated = generated.drop_duplicates("decoy_can")
    generated = generated.loc[~generated["decoy_can"].isin(excluded)].copy()

    rows = []
    for _, row in generated.iterrows():
        source_fp = fingerprint(row["source_can"])
        decoy_fp = fingerprint(row["decoy_can"])
        props = mol_props(row["decoy_can"])
        source_props = mol_props(row["source_can"])
        if source_fp is None or decoy_fp is None or props is None or source_props is None:
            continue
        sim = DataStructs.TanimotoSimilarity(source_fp, decoy_fp)
        if sim >= args.max_source_tanimoto:
            continue
        rows.append(
            {
                "SMILES": row["decoy_can"],
                "source_active": row["source_can"],
                "MW": props["MW"],
                "LogP": props["LogP"],
                "RB": props["RB"],
                "TPSA": props["TPSA"],
                "source_MW": source_props["MW"],
                "mw_diff_to_source": abs(props["MW"] - source_props["MW"]),
                "tanimoto_to_source": sim,
            }
        )
    candidates = pd.DataFrame(rows)
    if len(candidates) < target_n:
        raise RuntimeError("Only %d usable decoys for target %d" % (len(candidates), target_n))

    active_bins = (actives["MW"] / args.mw_bin_width).round().astype(int)
    need_by_bin = active_bins.value_counts().mul(args.ratio).to_dict()
    candidates["mw_bin"] = (candidates["MW"] / args.mw_bin_width).round().astype(int)
    candidates["jitter"] = [rng.random() for _ in range(len(candidates))]
    candidates = candidates.sort_values(["mw_diff_to_source", "tanimoto_to_source", "jitter"]).reset_index(drop=True)

    selected_parts = []
    used = set()
    for mw_bin, need in sorted(need_by_bin.items()):
        pool = candidates.loc[(candidates["mw_bin"].eq(mw_bin)) & (~candidates["SMILES"].isin(used))]
        if len(pool) < need:
            pool = candidates.loc[
                (candidates["mw_bin"].sub(mw_bin).abs() <= 1) & (~candidates["SMILES"].isin(used))
            ]
        take = pool.head(int(need))
        selected_parts.append(take)
        used |= set(take["SMILES"])

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if len(selected) < target_n:
        rest = candidates.loc[~candidates["SMILES"].isin(used)].head(target_n - len(selected))
        selected = pd.concat([selected, rest], ignore_index=True)
    selected = selected.drop_duplicates("SMILES").head(target_n).reset_index(drop=True)
    if len(selected) < target_n:
        raise RuntimeError("Selected only %d decoys for target %d" % (len(selected), target_n))

    decoys = pd.DataFrame(
        {
            "ID": ["%s_DeepCoy_global_test_decoy_%d" % (args.target, i) for i in range(len(selected))],
            "SMILES": selected["SMILES"],
            "Y": 0,
            "Protein": actives["Protein"].iloc[0],
            "target_cluster": actives["target_cluster"].iloc[0],
        }
    )
    out = pd.concat([actives[OUTPUT_COLUMNS], decoys[OUTPUT_COLUMNS]], ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_csv = args.outdir / ("%s_CHEMBL10_DeepCoy_global_MWmatched_1to10.csv" % args.target)
    pair_csv = args.outdir / ("%s_CHEMBL10_DeepCoy_global_selected_decoys.csv" % args.target)
    summary_json = args.outdir / "dataset_summary.json"
    out[OUTPUT_COLUMNS].to_csv(out_csv, index=False)
    selected.to_csv(pair_csv, index=False)

    summary = {
        "target": args.target,
        "strategy": "global_1to10_mw_distribution_matched",
        "test_csv": str(args.test_csv),
        "generated_pairs": str(args.generated_pairs),
        "n_actives": int(len(actives)),
        "n_decoys": int(len(decoys)),
        "ratio": "1:%d" % args.ratio,
        "usable_candidates": int(len(candidates)),
        "active_MW_median": float(actives["MW"].median()),
        "decoy_MW_median": float(selected["MW"].median()),
        "active_MW_q25": float(actives["MW"].quantile(0.25)),
        "active_MW_q75": float(actives["MW"].quantile(0.75)),
        "decoy_MW_q25": float(selected["MW"].quantile(0.25)),
        "decoy_MW_q75": float(selected["MW"].quantile(0.75)),
        "median_mw_diff_to_source": float(selected["mw_diff_to_source"].median()),
        "q75_mw_diff_to_source": float(selected["mw_diff_to_source"].quantile(0.75)),
        "max_source_tanimoto": args.max_source_tanimoto,
        "out_csv": str(out_csv),
        "selected_decoys": str(pair_csv),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
