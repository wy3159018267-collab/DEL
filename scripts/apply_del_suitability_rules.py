#!/usr/bin/env python3
"""Apply a unified DEL suitability rule set to target-level diagnostics.

The script reads a target-level summary table and produces:
  1. categorical diagnostic scores for each target,
  2. a pre-finetuning suitability decision,
  3. an observed/validated decision after model-gain results are available,
  4. a checklist heatmap and a ranked score plot.

The rule set is intentionally transparent and coarse. It is not a fitted model;
it is a reproducible decision framework for comparing DEL datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


STATUS_TEXT = {0: "poor", 1: "caution", 2: "good"}
STATUS_SYMBOL = {0: "-", 1: "+/-", 2: "+"}


def finite(value: object) -> bool:
    try:
        return np.isfinite(float(value))
    except Exception:
        return False


def fnum(row: pd.Series, col: str, default: float = np.nan) -> float:
    value = row.get(col, default)
    try:
        return float(value)
    except Exception:
        return default


def raw_signal_score(row: pd.Series) -> tuple[int, str]:
    level = str(row.get("raw_signal_level", "")).lower()
    note = str(row.get("raw_signal_note", ""))
    if "strong" in level:
        return 2, "Raw DEL signal is strong/auditable. " + note
    if "available" in level or "copies" in level or "positive-only" in level:
        return 1, "Raw signal is partly available or externally defined. " + note
    return 0, "Raw enrichment/readout is not auditable from supplied data. " + note


def model_response_score(row: pd.Series) -> tuple[int, str]:
    clear = fnum(row, "clear_gain_models_both_EF1_AUPRC")
    if not finite(clear):
        return 1, "Model response unavailable; treated as caution."
    if clear >= 6:
        return 2, f"{int(clear)}/7 models show both EF1 and AUPRC gain."
    if clear >= 3:
        return 1, f"{int(clear)}/7 models show both EF1 and AUPRC gain."
    return 0, f"Only {int(clear)}/7 models show both EF1 and AUPRC gain."


def pos_neg_boundary_score(row: pd.Series) -> tuple[int, str]:
    x = fnum(row, "pos_neg_nearest_median")
    if not finite(x):
        return 1, "Positive-negative nearest similarity unavailable."
    if x >= 0.40:
        return 2, f"Positive-negative boundary is non-trivial (median NN={x:.3f})."
    if x >= 0.30:
        return 1, f"Positive-negative boundary is intermediate (median NN={x:.3f})."
    return 0, f"Positive-negative boundary is easy/decoy-like (median NN={x:.3f})."


def del_test_coverage_score(row: pd.Series) -> tuple[int, str]:
    x = fnum(row, "test_active_to_DEL_positive_median")
    if not finite(x):
        return 1, "Test-active to DEL-positive coverage unavailable."
    if x >= 0.28:
        return 2, f"DEL positives moderately cover test actives (median NN={x:.3f})."
    if x >= 0.24:
        return 1, f"DEL-test coverage is limited/intermediate (median NN={x:.3f})."
    return 0, f"DEL-test coverage is weak (median NN={x:.3f})."


def physchem_match_score(row: pd.Series) -> tuple[int, str]:
    mw = abs(fnum(row, "MW_median_shift_trainpos_minus_testactive", 0.0))
    tpsa = abs(fnum(row, "TPSA_median_shift_trainpos_minus_testactive", 0.0))
    rotb = abs(fnum(row, "RotB_median_shift_trainpos_minus_testactive", 0.0))
    if mw <= 35 and tpsa <= 30 and rotb <= 4:
        return 2, f"Physchem shift is small (MW={mw:.1f}, TPSA={tpsa:.1f}, RotB={rotb:.1f})."
    if mw <= 90 and tpsa <= 50 and rotb <= 5:
        return 1, f"Physchem shift is moderate (MW={mw:.1f}, TPSA={tpsa:.1f}, RotB={rotb:.1f})."
    return 0, f"Physchem shift is large (MW={mw:.1f}, TPSA={tpsa:.1f}, RotB={rotb:.1f})."


def leakage_risk_score(row: pd.Series) -> tuple[int, str]:
    tp = fnum(row, "train_positive_test_active_exact_overlap", 0.0)
    tn = fnum(row, "train_negative_test_decoy_exact_overlap", 0.0)
    if tp > 0:
        return 0, f"Train positives overlap test actives (n={int(tp)})."
    if tn > 500:
        return 1, f"Train negatives overlap test decoys substantially (n={int(tn)})."
    return 2, f"No train-positive/test-active overlap; decoy overlap acceptable (n={int(tn)})."


def prefit_decision(score: int) -> str:
    if score >= 8:
        return "recommended"
    if score >= 5:
        return "conditional"
    return "not_recommended"


def observed_decision(prefit_score: int, model_score: int) -> str:
    if model_score == 2 and prefit_score >= 5:
        return "validated_suitable"
    if model_score == 2 and prefit_score < 5:
        return "works_but_needs_data_audit"
    if model_score == 1:
        return "model_dependent"
    return "not_validated_or_problematic"


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scorers = [
        ("raw_signal", raw_signal_score, True),
        ("pos_neg_boundary", pos_neg_boundary_score, True),
        ("del_test_coverage", del_test_coverage_score, True),
        ("physchem_match", physchem_match_score, True),
        ("low_leakage_risk", leakage_risk_score, True),
        ("cross_model_gain", model_response_score, False),
    ]
    for _, row in df.iterrows():
        out = {
            "target": row["target"],
            "family": row.get("family", ""),
            "original_final_class": row.get("final_class", ""),
        }
        reasons = []
        prefit = 0
        for name, scorer, is_prefit in scorers:
            score, reason = scorer(row)
            out[name + "_score"] = score
            out[name + "_status"] = STATUS_TEXT[score]
            out[name + "_reason"] = reason
            reasons.append(f"{name}: {reason}")
            if is_prefit:
                prefit += score
        model_score = int(out["cross_model_gain_score"])
        # Raw DEL signal is the anchor of a DEL-derived training set, so it
        # receives double weight in the pre-finetuning rule. This prevents a
        # strong, auditable DEL signal from being overruled by one imperfect
        # transfer diagnostic.
        weighted_prefit = prefit + int(out["raw_signal_score"])
        out["prefit_score_0to12"] = weighted_prefit
        out["observed_score_0to14"] = weighted_prefit + model_score
        out["prefit_decision"] = prefit_decision(weighted_prefit)
        out["observed_decision"] = observed_decision(weighted_prefit, model_score)
        out["decision_reasons"] = " | ".join(reasons)
        rows.append(out)
    return pd.DataFrame(rows)


def plot_checklist(rule_df: pd.DataFrame, output_dir: Path) -> None:
    score_cols = [
        "raw_signal_score",
        "pos_neg_boundary_score",
        "del_test_coverage_score",
        "physchem_match_score",
        "low_leakage_risk_score",
        "cross_model_gain_score",
    ]
    label_cols = [
        "Raw DEL signal",
        "Pos-neg boundary",
        "DEL-test coverage",
        "Physchem match",
        "Low leakage risk",
        "Cross-model gain",
    ]
    mat = rule_df.set_index("target")[score_cols].astype(int)
    mat.columns = label_cols
    cmap = ListedColormap(["#d73027", "#fee08b", "#1a9850"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.imshow(mat.values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, STATUS_TEXT[int(mat.iloc[i, j])], ha="center", va="center", fontsize=8)
    ax.set_title("Unified DEL suitability rule checklist")
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(
        handles=[
            Patch(facecolor="#1a9850", label="good"),
            Patch(facecolor="#fee08b", label="caution"),
            Patch(facecolor="#d73027", label="poor"),
        ],
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "unified_DEL_suitability_rule_checklist.png", dpi=300)
    fig.savefig(output_dir / "unified_DEL_suitability_rule_checklist.pdf")
    plt.close(fig)


def plot_ranked(rule_df: pd.DataFrame, output_dir: Path) -> None:
    decision_palette = {
        "recommended": "#1a9850",
        "conditional": "#fee08b",
        "not_recommended": "#d73027",
    }
    ranked = rule_df.sort_values("prefit_score_0to12", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [decision_palette[d] for d in ranked["prefit_decision"]]
    ax.bar(ranked["target"], ranked["prefit_score_0to12"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Pre-finetuning rule score (0-12)")
    ax.set_xlabel("Target")
    ax.set_title("DEL dataset suitability before fine-tuning")
    ax.set_ylim(0, 12)
    ax.tick_params(axis="x", rotation=30)
    for i, row in enumerate(ranked.itertuples(index=False)):
        ax.text(i, row.prefit_score_0to12 + 0.2, str(int(row.prefit_score_0to12)), ha="center", fontsize=9)
    ax.legend(
        handles=[Patch(facecolor=v, label=k) for k, v in decision_palette.items()],
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "unified_DEL_prefit_score_ranked.png", dpi=300)
    fig.savefig(output_dir / "unified_DEL_prefit_score_ranked.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="all9_DEL_suitability_master_table.csv",
        help="Target-level master table.",
    )
    parser.add_argument(
        "--output-dir",
        default="rule_based_DEL_suitability",
        help="Directory for rule outputs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    rule_df = apply_rules(df)

    # Preserve target order from input.
    rule_df["target"] = pd.Categorical(rule_df["target"], categories=df["target"].astype(str).tolist(), ordered=True)
    rule_df = rule_df.sort_values("target").reset_index(drop=True)
    rule_df["target"] = rule_df["target"].astype(str)

    rule_df.to_csv(output_dir / "DEL_suitability_rule_scores.csv", index=False)

    symbolic = rule_df[[
        "target",
        "raw_signal_score",
        "pos_neg_boundary_score",
        "del_test_coverage_score",
        "physchem_match_score",
        "low_leakage_risk_score",
        "cross_model_gain_score",
        "prefit_score_0to12",
        "observed_score_0to14",
        "prefit_decision",
        "observed_decision",
    ]].copy()
    for col in [c for c in symbolic.columns if c.endswith("_score")]:
        symbolic[col.replace("_score", "_symbol")] = symbolic[col].map(STATUS_SYMBOL)
    symbolic.to_csv(output_dir / "DEL_suitability_rule_symbolic_summary.csv", index=False)

    plot_checklist(rule_df, output_dir)
    plot_ranked(rule_df, output_dir)

    readme = output_dir / "README_rules.md"
    readme.write_text(
        "# Unified DEL Suitability Rule Set\n\n"
        "This directory was generated by `apply_del_suitability_rules.py`.\n\n"
        "The pre-finetuning score uses five data-level diagnostics: raw DEL signal, "
        "positive-negative boundary, DEL-test coverage, physicochemical match, and "
        "leakage risk. The observed score additionally includes cross-model fine-tuning gain.\n\n"
        "Decision thresholds:\n"
        "- prefit score >= 8: recommended\n"
        "- prefit score 5-7: conditional\n"
        "- prefit score <= 4: not_recommended\n\n"
        "The rules are intentionally interpretable and should be treated as a diagnostic "
        "framework, not as a statistically fitted predictor.\n",
        encoding="utf-8",
    )

    print(f"Wrote {output_dir}")
    print(rule_df[["target", "prefit_score_0to12", "prefit_decision", "observed_score_0to14", "observed_decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
