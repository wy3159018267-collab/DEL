#!/usr/bin/env python3
"""Select DeepCoy candidates with MW-first matching for DEL fine-tuning decoys."""

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
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
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
            if len(toks) < 2:
                continue
            rows.append({"active_smiles": toks[0], "decoy_smiles": toks[1]})
    if not rows:
        raise RuntimeError(f"No generated pairs found in {path}")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-train", required=True, type=Path)
    parser.add_argument("--reference-val", required=True, type=Path)
    parser.add_argument("--generated-pairs", required=True, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-tanimoto", type=float, default=0.3)
    parser.add_argument("--mw-tolerance", type=float, default=50.0)
    parser.add_argument("--logp-tolerance", type=float, default=1.0)
    parser.add_argument("--rb-tolerance", type=int, default=4)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train = pd.read_csv(args.reference_train)
    val = pd.read_csv(args.reference_val)
    train_pos = train.loc[train["Y"].eq(1), OUTPUT_COLUMNS].copy().reset_index(drop=True)
    val_pos = val.loc[val["Y"].eq(1), OUTPUT_COLUMNS].copy().reset_index(drop=True)
    positives = pd.concat([train_pos.assign(split="train"), val_pos.assign(split="val")], ignore_index=True)
    positives["active_can"] = positives["SMILES"].map(canonical_smiles)
    positives["active_props"] = positives["active_can"].map(mol_props)
    positives["active_fp"] = positives["active_can"].map(fingerprint)
    positives = positives.dropna(subset=["active_can", "active_props", "active_fp"]).reset_index(drop=True)

    test = pd.read_csv(args.test_csv)
    test_can = set(test["SMILES"].dropna().map(canonical_smiles).dropna()) if "SMILES" in test.columns else set()
    positive_can = set(positives["active_can"])
    generated = read_generated_pairs(args.generated_pairs)
    generated["active_can"] = generated["active_smiles"].map(canonical_smiles)
    generated["decoy_can"] = generated["decoy_smiles"].map(canonical_smiles)
    generated = generated.dropna(subset=["active_can", "decoy_can"]).drop_duplicates(["active_can", "decoy_can"])
    generated = generated.loc[~generated["decoy_can"].isin(test_can | positive_can)].copy()

    train_decoys = []
    val_decoys = []
    selected_rows = []
    missing = []
    used_decoys = set()

    for _, active in positives.iterrows():
        active_props = active["active_props"]
        active_fp = active["active_fp"]
        candidates = generated.loc[generated["active_can"].eq(active["active_can"]), "decoy_can"].tolist()
        rng.shuffle(candidates)
        scored = []
        for decoy in candidates:
            if decoy in used_decoys:
                continue
            decoy_props = mol_props(decoy)
            decoy_fp = fingerprint(decoy)
            if decoy_props is None or decoy_fp is None:
                continue
            sim = DataStructs.TanimotoSimilarity(active_fp, decoy_fp)
            if sim >= args.max_tanimoto:
                continue
            mw_diff = abs(decoy_props["MW"] - active_props["MW"])
            logp_diff = abs(decoy_props["LogP"] - active_props["LogP"])
            rb_diff = abs(decoy_props["RB"] - active_props["RB"])
            strict = (
                mw_diff <= args.mw_tolerance
                and logp_diff <= args.logp_tolerance
                and rb_diff <= args.rb_tolerance
            )
            score = (
                mw_diff / max(args.mw_tolerance, 1e-9)
                + 0.25 * logp_diff / max(args.logp_tolerance, 1e-9)
                + 0.10 * rb_diff / max(args.rb_tolerance, 1e-9)
                + 0.10 * sim
            )
            scored.append((not strict, score, mw_diff, logp_diff, rb_diff, sim, decoy, decoy_props))
        scored.sort(key=lambda x: (x[0], x[1], x[2]))
        chosen = scored[: args.ratio]
        if len(chosen) < args.ratio:
            missing.append({"active_smiles": active["SMILES"], "split": active["split"], "n_decoys": len(chosen)})
            continue
        target_rows = train_decoys if active["split"] == "train" else val_decoys
        for _, score, mw_diff, logp_diff, rb_diff, sim, decoy, decoy_props in chosen:
            used_decoys.add(decoy)
            target_rows.append(decoy)
            selected_rows.append(
                {
                    "active_smiles": active["active_can"],
                    "decoy_smiles": decoy,
                    "split": active["split"],
                    "score": score,
                    "mw_diff": mw_diff,
                    "logp_diff": logp_diff,
                    "rb_diff": rb_diff,
                    "tanimoto_to_active": sim,
                    "active_MW": active_props["MW"],
                    "decoy_MW": decoy_props["MW"],
                    "active_LogP": active_props["LogP"],
                    "decoy_LogP": decoy_props["LogP"],
                    "active_RB": active_props["RB"],
                    "decoy_RB": decoy_props["RB"],
                }
            )

    if missing:
        missing_path = args.outdir / f"{args.target}_missing_deepcoy_mw_matched_decoys.csv"
        pd.DataFrame(missing).to_csv(missing_path, index=False)
        raise RuntimeError(f"{len(missing)} positives have fewer than {args.ratio} selected decoys; see {missing_path}")

    protein = train_pos["Protein"].iloc[0]
    target_cluster = train_pos["target_cluster"].iloc[0]

    def make_neg_rows(smiles, prefix):
        return pd.DataFrame(
            {
                "ID": [f"{prefix}_{i}" for i in range(len(smiles))],
                "SMILES": smiles,
                "Y": 0,
                "Protein": protein,
                "target_cluster": target_cluster,
            }
        )

    train_neg = make_neg_rows(train_decoys, f"{args.target}_DeepCoy_MW_train_decoy")
    val_neg = make_neg_rows(val_decoys, f"{args.target}_DeepCoy_MW_val_decoy")
    out_train = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=args.seed)
    out_val = pd.concat([val_pos, val_neg], ignore_index=True).sample(frac=1, random_state=args.seed)
    out_train[OUTPUT_COLUMNS].to_csv(args.outdir / "train.csv", index=False)
    out_val[OUTPUT_COLUMNS].to_csv(args.outdir / "val.csv", index=False)

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.outdir / f"{args.target}_selected_deepcoy_mw_matched_pairs.csv", index=False)
    summary = {
        "target": args.target,
        "generated_pairs": str(args.generated_pairs),
        "test_exclusion_source": str(args.test_csv),
        "ratio": f"1:{args.ratio}",
        "max_tanimoto": args.max_tanimoto,
        "mw_tolerance_priority": args.mw_tolerance,
        "logp_tolerance_priority": args.logp_tolerance,
        "rb_tolerance_priority": args.rb_tolerance,
        "train_actives": int(out_train["Y"].eq(1).sum()),
        "train_decoys": int(out_train["Y"].eq(0).sum()),
        "val_actives": int(out_val["Y"].eq(1).sum()),
        "val_decoys": int(out_val["Y"].eq(0).sum()),
        "selected_decoys": int(len(selected)),
        "median_mw_diff": float(selected["mw_diff"].median()),
        "q75_mw_diff": float(selected["mw_diff"].quantile(0.75)),
        "max_mw_diff": float(selected["mw_diff"].max()),
        "train_csv": str(args.outdir / "train.csv"),
        "val_csv": str(args.outdir / "val.csv"),
    }
    (args.outdir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
