#!/usr/bin/env python3
"""Prepare a stratified subset of test actives for global DeepCoy test decoys."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors


RDLogger.DisableLog("rdApp.*")


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


def mol_wt(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Descriptors.MolWt(mol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--n-seeds", type=int, default=3200)
    parser.add_argument("--n-chunks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(args.test_csv)
    pos = test.loc[test["Y"].eq(1)].copy()
    pos["canonical_smiles"] = pos["SMILES"].map(canonical_smiles)
    pos = pos.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles")
    pos["MW"] = pos["canonical_smiles"].map(mol_wt)
    pos = pos.dropna(subset=["MW"]).reset_index(drop=True)

    n_seeds = min(args.n_seeds, len(pos))
    pos["mw_bin"] = pd.qcut(pos["MW"], q=min(20, len(pos)), duplicates="drop")
    sampled = (
        pos.groupby("mw_bin", observed=True, group_keys=False)
        .apply(lambda x: x.sample(n=max(1, round(n_seeds * len(x) / len(pos))), random_state=args.seed))
        .drop_duplicates("canonical_smiles")
    )
    if len(sampled) > n_seeds:
        sampled = sampled.sample(n=n_seeds, random_state=args.seed)
    elif len(sampled) < n_seeds:
        rest = pos.loc[~pos["canonical_smiles"].isin(sampled["canonical_smiles"])]
        sampled = pd.concat(
            [sampled, rest.sample(n=min(n_seeds - len(sampled), len(rest)), random_state=args.seed)],
            ignore_index=True,
        )
    sampled = sampled.sort_values("MW").reset_index(drop=True)
    sampled.to_csv(args.outdir / ("%s_global_seed_actives.csv" % args.prefix), index=False)

    counts = []
    for idx in range(args.n_chunks):
        chunk = sampled.iloc[idx :: args.n_chunks].copy()
        chunk_path = args.outdir / ("%s_global_seed_chunk_%02d.smi" % (args.prefix, idx))
        chunk["canonical_smiles"].to_csv(chunk_path, index=False, header=False)
        counts.append({"chunk": idx, "path": str(chunk_path), "n_seed_actives": int(len(chunk))})

    summary = {
        "prefix": args.prefix,
        "test_csv": str(args.test_csv),
        "n_test_positives": int(test["Y"].eq(1).sum()),
        "n_unique_valid_positives": int(len(pos)),
        "n_seed_actives": int(len(sampled)),
        "n_chunks": args.n_chunks,
        "seed_mw_median": float(sampled["MW"].median()),
        "test_active_mw_median": float(pos["MW"].median()),
        "chunks": counts,
    }
    (args.outdir / ("%s_global_seed_input_summary.json" % args.prefix)).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
