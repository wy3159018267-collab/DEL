#!/usr/bin/env python3
"""Prepare unique DEL positives as DeepCoy active input chunks."""

# from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-train", required=True, type=Path)
    parser.add_argument("--reference-val", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--n-chunks", type=int, default=8)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.reference_train)
    val = pd.read_csv(args.reference_val)
    for path, df in [(args.reference_train, train), (args.reference_val, val)]:
        missing = set(OUTPUT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {missing}")

    pos = pd.concat(
        [
            train.loc[train["Y"].eq(1), OUTPUT_COLUMNS].assign(split="train"),
            val.loc[val["Y"].eq(1), OUTPUT_COLUMNS].assign(split="val"),
        ],
        ignore_index=True,
    )
    pos["canonical_smiles"] = pos["SMILES"].map(canonical_smiles)
    pos = pos.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles").reset_index(drop=True)
    pos.to_csv(args.outdir / f"{args.prefix}_DEL_positives_for_deepcoy.csv", index=False)

    counts = []
    for idx in range(args.n_chunks):
        chunk = pos.iloc[idx :: args.n_chunks].copy()
        chunk_path = args.outdir / f"{args.prefix}_active_chunk_{idx:02d}.smi"
        chunk["canonical_smiles"].to_csv(chunk_path, index=False, header=False)
        counts.append({"chunk": idx, "path": str(chunk_path), "n_actives": int(len(chunk))})

    summary = {
        "prefix": args.prefix,
        "reference_train": str(args.reference_train),
        "reference_val": str(args.reference_val),
        "n_train_positives": int(train["Y"].eq(1).sum()),
        "n_val_positives": int(val["Y"].eq(1).sum()),
        "n_unique_valid_positives": int(len(pos)),
        "n_chunks": args.n_chunks,
        "chunks": counts,
    }
    (args.outdir / f"{args.prefix}_deepcoy_input_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
