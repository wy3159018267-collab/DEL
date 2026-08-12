#!/usr/bin/env python3
"""Build public-positive + DUD-E-decoy fine-tuning train/val sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")

OUTPUT_COLUMNS = ["ID", "SMILES", "Y", "Protein", "target_cluster"]
TOLERANCE = {
    "MW": 50.0,
    "LogP": 0.5,
    "HBA": 2,
    "HBD": 1,
    "RB": 2,
}


def canonical_smiles(smi: str) -> str | None:
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


def get_properties(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RB": Descriptors.NumRotatableBonds(mol),
    }


def get_fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def load_label_counts(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path)
    if "Y" not in df.columns:
        raise ValueError(f"{path} lacks Y column")
    return int(df["Y"].eq(1).sum()), int(df["Y"].eq(0).sum())


def load_reference_metadata(path: Path) -> tuple[str, int | str]:
    df = pd.read_csv(path)
    missing = set(["Protein", "target_cluster"]) - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks metadata columns: {missing}")
    return df["Protein"].iloc[0], df["target_cluster"].iloc[0]


def build_active_reference(positive_smiles: list[str]):
    active_props = []
    active_fps = []
    active_canonical = set()
    for smi in positive_smiles:
        can = canonical_smiles(smi)
        if can is None:
            continue
        mol = Chem.MolFromSmiles(can)
        if mol is None:
            continue
        props = get_properties(mol)
        props["SMILES"] = smi
        active_props.append(props)
        active_fps.append(get_fingerprint(mol))
        active_canonical.add(can)
    if not active_props:
        raise RuntimeError("No valid public positive molecules found.")
    global_bounds = {
        "MW": [
            min(p["MW"] for p in active_props) - TOLERANCE["MW"],
            max(p["MW"] for p in active_props) + TOLERANCE["MW"],
        ],
        "LogP": [
            min(p["LogP"] for p in active_props) - TOLERANCE["LogP"],
            max(p["LogP"] for p in active_props) + TOLERANCE["LogP"],
        ],
    }
    return active_canonical, active_props, active_fps, global_bounds


def is_dude_like_decoy(
    smi: str,
    excluded_canonical: set[str],
    active_props: list[dict],
    active_fps,
    global_bounds: dict,
    max_tanimoto: float,
) -> bool:
    can = canonical_smiles(smi)
    if can is None or can in excluded_canonical:
        return False
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return False
    mw = Descriptors.MolWt(mol)
    if not (global_bounds["MW"][0] <= mw <= global_bounds["MW"][1]):
        return False
    logp = Descriptors.MolLogP(mol)
    if not (global_bounds["LogP"][0] <= logp <= global_bounds["LogP"][1]):
        return False
    props = get_properties(mol)
    matched = False
    for active in active_props:
        if (
            abs(props["MW"] - active["MW"]) <= TOLERANCE["MW"]
            and abs(props["LogP"] - active["LogP"]) <= TOLERANCE["LogP"]
            and abs(props["HBA"] - active["HBA"]) <= TOLERANCE["HBA"]
            and abs(props["HBD"] - active["HBD"]) <= TOLERANCE["HBD"]
            and abs(props["RB"] - active["RB"]) <= TOLERANCE["RB"]
        ):
            matched = True
            break
    if not matched:
        return False
    fp = get_fingerprint(mol)
    if max(DataStructs.BulkTanimotoSimilarity(fp, active_fps)) >= max_tanimoto:
        return False
    return True


def collect_decoys(
    dude_pool: Path,
    n_needed: int,
    excluded_canonical: set[str],
    active_props: list[dict],
    active_fps,
    global_bounds: dict,
    max_tanimoto: float,
    seed: int,
) -> tuple[list[str], int]:
    dude = pd.read_csv(dude_pool)
    if "SMILES" not in dude.columns:
        raise ValueError(f"{dude_pool} lacks SMILES column")
    if "Y" in dude.columns:
        active_in_dude = set(dude.loc[dude["Y"].eq(1), "SMILES"].dropna().map(canonical_smiles).dropna())
        pool = dude.loc[dude["Y"].eq(0), "SMILES"].dropna().drop_duplicates().tolist()
        excluded_canonical = set(excluded_canonical) | active_in_dude
    else:
        pool = dude["SMILES"].dropna().drop_duplicates().tolist()
    random.Random(seed).shuffle(pool)

    selected = []
    selected_can = set()
    checked = 0
    for smi in tqdm(pool, desc="Filtering DUD-E total pool"):
        if len(selected) >= n_needed:
            break
        checked += 1
        can = canonical_smiles(smi)
        if can is None or can in selected_can:
            continue
        if is_dude_like_decoy(smi, excluded_canonical | selected_can, active_props, active_fps, global_bounds, max_tanimoto):
            selected.append(smi)
            selected_can.add(can)
    if len(selected) < n_needed:
        raise RuntimeError(f"Only found {len(selected)} DUD-E decoys, but need {n_needed}.")
    return selected, checked


def make_rows(smiles: list[str], prefix: str, y: int, protein: str, target_cluster) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [f"{prefix}_{i}" for i in range(len(smiles))],
            "SMILES": smiles,
            "Y": y,
            "Protein": protein,
            "target_cluster": target_cluster,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--public-csv", required=True, type=Path)
    parser.add_argument("--dude-pool", required=True, type=Path)
    parser.add_argument("--reference-train", required=True, type=Path)
    parser.add_argument("--reference-val", required=True, type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-tanimoto", type=float, default=0.3)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    train_pos_n, _ = load_label_counts(args.reference_train)
    val_pos_n, _ = load_label_counts(args.reference_val)
    total_pos_n = train_pos_n + val_pos_n
    protein, target_cluster = load_reference_metadata(args.reference_train)

    public = pd.read_csv(args.public_csv)
    if "SMILES" not in public.columns or "Y" not in public.columns:
        raise ValueError("Public CSV must contain SMILES and Y columns")
    public_pos = public[public["Y"].eq(1)].dropna(subset=["SMILES"]).copy()
    public_pos["canonical_smiles"] = public_pos["SMILES"].map(canonical_smiles)
    public_pos = public_pos.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles")
    if len(public_pos) < total_pos_n:
        raise RuntimeError(f"Only {len(public_pos)} public positives after canonical dedup; need {total_pos_n}.")

    sampled_pos = public_pos.sample(n=total_pos_n, random_state=args.seed).reset_index(drop=True)
    train_pos_smiles = sampled_pos.iloc[:train_pos_n]["SMILES"].tolist()
    val_pos_smiles = sampled_pos.iloc[train_pos_n:]["SMILES"].tolist()
    selected_positive_smiles = train_pos_smiles + val_pos_smiles
    active_canonical, active_props, active_fps, global_bounds = build_active_reference(selected_positive_smiles)

    excluded = set(active_canonical)
    if args.test_csv is not None:
        test = pd.read_csv(args.test_csv)
        if "SMILES" in test.columns:
            excluded |= set(test["SMILES"].dropna().map(canonical_smiles).dropna())

    total_decoys = total_pos_n * args.ratio
    decoys, dude_checked = collect_decoys(
        args.dude_pool,
        total_decoys,
        excluded,
        active_props,
        active_fps,
        global_bounds,
        args.max_tanimoto,
        args.seed,
    )

    train_pos = make_rows(train_pos_smiles, f"{args.target}_public_train_active", 1, protein, target_cluster)
    val_pos = make_rows(val_pos_smiles, f"{args.target}_public_val_active", 1, protein, target_cluster)
    train_decoys = make_rows(decoys[: train_pos_n * args.ratio], f"{args.target}_DUDE_train_decoy", 0, protein, target_cluster)
    val_decoys = make_rows(decoys[train_pos_n * args.ratio :], f"{args.target}_DUDE_val_decoy", 0, protein, target_cluster)

    train = pd.concat([train_pos, train_decoys], ignore_index=True).sample(frac=1.0, random_state=args.seed)
    val = pd.concat([val_pos, val_decoys], ignore_index=True).sample(frac=1.0, random_state=args.seed)
    train[OUTPUT_COLUMNS].to_csv(args.outdir / "train.csv", index=False)
    val[OUTPUT_COLUMNS].to_csv(args.outdir / "val.csv", index=False)
    sampled_pos.to_csv(args.outdir / f"{args.target}_selected_public_positives.csv", index=False)
    pd.DataFrame({"SMILES": decoys}).to_csv(args.outdir / f"{args.target}_generated_DUDE_decoys.csv", index=False)

    summary = {
        "target": args.target,
        "public_positive_source": str(args.public_csv),
        "dude_total_pool": str(args.dude_pool),
        "reference_train": str(args.reference_train),
        "reference_val": str(args.reference_val),
        "test_exclusion_source": str(args.test_csv) if args.test_csv else None,
        "ratio": f"1:{args.ratio}",
        "seed": args.seed,
        "max_tanimoto_to_public_positive": args.max_tanimoto,
        "available_public_positives_after_canonical_dedup": int(len(public_pos)),
        "selected_public_positives": int(total_pos_n),
        "train_actives": int(train["Y"].eq(1).sum()),
        "train_decoys": int(train["Y"].eq(0).sum()),
        "val_actives": int(val["Y"].eq(1).sum()),
        "val_decoys": int(val["Y"].eq(0).sum()),
        "generated_dude_decoys": int(len(decoys)),
        "dude_pool_rows_checked": int(dude_checked),
        "train_csv": str(args.outdir / "train.csv"),
        "val_csv": str(args.outdir / "val.csv"),
        "rule": (
            "Public positives sampled after removing test-set leakage; negatives generated from the total DUD-E pool, "
            "restricted to DUD-E Y=0 when labels are available, excluding DUD-E Y=1, selected positives, and test-set SMILES; "
            "DUD-E-like property tolerances match the existing PARP2 decoy-generation script."
        ),
    }
    (args.outdir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
