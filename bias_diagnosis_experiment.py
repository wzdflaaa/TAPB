import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

BRANCHES = [
    "factual",
    "cf_drug",
    "cf_protein",
    "debiased_drug_only",
    "debiased_protein_only",
    "debiased",
]

REQUIRED_KEYS = {
    "label",
    "factual_logits",
    "cf_drug_logits",
    "cf_protein_logits",
    "debiased_drug_only_logits",
    "debiased_protein_only_logits",
    "debiased_logits",
    "pr_id",
    "SMILES",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: bias diagnosis")
    parser.add_argument("--pt", required=True, type=str, help="path to test_predictions.pt")
    parser.add_argument("--train_csv", type=str, default=None, help="path to train csv")
    parser.add_argument("--test_csv", type=str, default=None, help="path to test csv")
    parser.add_argument("--data_dir", type=str, default=None, help="dataset split directory, e.g. datasets/bindingdb/cluster")
    parser.add_argument("--split", type=str, default=None, choices=["random", "cold", "cluster", "augmented"], help="split name for auto csv resolution")
    parser.add_argument("--output_dir", required=True, type=str, help="output directory")
    return parser.parse_args()

def resolve_csv_paths(args) -> Tuple[str, str]:
    if args.train_csv and args.test_csv:
        return args.train_csv, args.test_csv
    if args.data_dir is None or args.split is None:
        raise ValueError(
            "Either provide both --train_csv/--test_csv, or provide --data_dir with --split."
        )
    if args.split == "cluster":
        train_csv = os.path.join(args.data_dir, "source_train_with_id.csv")
        test_csv = os.path.join(args.data_dir, "target_test_with_id.csv")
    else:
        train_csv = os.path.join(args.data_dir, "train_with_id.csv")
        test_csv = os.path.join(args.data_dir, "test_with_id.csv")
    for path in (train_csv, test_csv):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
    return train_csv, test_csv

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pick_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of columns found: {candidates}")


def _corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    return {"pearson": pearson, "spearman": spearman}


def validate_and_load_pt(pt_path: str) -> Dict[str, list]:
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError("PT file should be a dict")
    missing = REQUIRED_KEYS - set(data.keys())
    if missing:
        raise KeyError(f"PT file missing keys: {sorted(missing)}")
    n = len(data["label"])
    for k in REQUIRED_KEYS:
        if len(data[k]) != n:
            raise ValueError(f"Length mismatch: key={k}, expected={n}, got={len(data[k])}")
    return data


def build_group_prior(train_df: pd.DataFrame, group_col: str, label_col: str = "Y") -> pd.DataFrame:
    tmp = train_df[[group_col, label_col]].copy()
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce").fillna(0).astype(int)
    tmp[label_col] = (tmp[label_col] > 0).astype(int)
    return tmp.groupby(group_col, as_index=False)[label_col].mean().rename(columns={label_col: "prior"})


def validate_test_alignment(pred_df: pd.DataFrame, test_df: pd.DataFrame):
    if len(pred_df) != len(test_df):
        raise ValueError(
            f"Length mismatch between pred_df ({len(pred_df)}) and test_csv ({len(test_df)})."
        )

    test_pr_col = _pick_col(test_df, ["pr_id", "target_id", "Target", "Protein"])
    test_drug_col = _pick_col(test_df, ["SMILES", "Drug", "drug_id"])

    same_pr = np.array_equal(pred_df["pr_id"].astype(str).to_numpy(), test_df[test_pr_col].astype(str).to_numpy())
    same_drug = np.array_equal(pred_df["SMILES"].astype(str).to_numpy(), test_df[test_drug_col].astype(str).to_numpy())

    return {
        "same_length": True,
        "same_pr_id_order": bool(same_pr),
        "same_drug_order": bool(same_drug),
        "test_pr_col": test_pr_col,
        "test_drug_col": test_drug_col,
    }


def plot_branch_hist(pred_df: pd.DataFrame, kind:str,output_png: str):
    # branches = ["factual", "cf_drug", "cf_protein", "debiased"]
    n_branches=len(BRANCHES)
    ncols=3
    nrows=int(np.ceil(n_branches/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,4*nrows))
    axes=np.atleast_1d(axes).flatten()
    for ax, branch in zip(axes, BRANCHES):
        ax.hist(pred_df[f"{branch}_{kind}"], bins=40, alpha=0.8)
        ax.set_title(f"{branch} {kind} distribution")
        ax.set_xlabel("kind")
        ax.set_ylabel("count")
    for ax in axes[len(BRANCHES):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_group_hist(merged: pd.DataFrame, group_name: str, output_png: str):
    fcols = ["prior"] + [f"{branch}_prob" for branch in BRANCHES]
    ncols = 3
    nrows = int(np.ceil(len(ncols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, ncols):
        ax.hist(merged[col], bins=30, alpha=0.8)
        ax.set_title(f"{group_name} {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    for ax in axes[len(ncols):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_scatter(x, y, title, xlabel, ylabel, output_png):
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(x, y, s=18, alpha=0.7)
    if len(x) >= 2 and np.std(x) > 0:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        xs = np.linspace(np.min(x), np.max(x), 100)
        plt.plot(xs, p(xs), linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def compute_pos_neg_stats(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    #branches = ["factual", "cf_drug", "cf_protein", "debiased"]
    for branch in BRANCHES:
        for kind in ["logits", "prob"]:
            col = f"{branch}_{kind}"
            pos = pred_df.loc[pred_df["label"] == 1, col].to_numpy()
            neg = pred_df.loc[pred_df["label"] == 0, col].to_numpy()
            rows.append(
                {
                    "branch": branch,
                    "type": kind,
                    "pos_count": int(len(pos)),
                    "neg_count": int(len(neg)),
                    "pos_mean": float(np.mean(pos)) if len(pos) else np.nan,
                    "pos_std": float(np.std(pos)) if len(pos) else np.nan,
                    "neg_mean": float(np.mean(neg)) if len(neg) else np.nan,
                    "neg_std": float(np.std(neg)) if len(neg) else np.nan,
                    "mean_gap": float(np.mean(pos) - np.mean(neg)) if len(pos) and len(neg) else np.nan,
                }
            )
    return pd.DataFrame(rows)

def build_prediction_dataframe(data: Dict[str, list]) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "label": np.asarray(data["label"], dtype=int),
            "pr_id": data["pr_id"],
            "SMILES": data["SMILES"],
        }
    )
    for branch in BRANCHES:
        logits_key = f"{branch}_logits"
        pred_df[logits_key] = np.asarray(data[logits_key], dtype=float)
        pred_df[f"{branch}_prob"] = sigmoid(pred_df[logits_key].to_numpy())
    return pred_df


def summarize_prior_correlations(merged: pd.DataFrame, group_name: str) -> Dict[str, Dict[str, float]]:
    summary = {
        f"{group_name}_prior_vs_{branch}": _corr(
            merged["prior"].to_numpy(),
            merged[f"{branch}_prob"].to_numpy(),
        )
        for branch in BRANCHES
    }
    summary["n_overlap_groups"] = int(len(merged))
    return summary


def export_scatter_plots(merged: pd.DataFrame, group_name: str, output_dir: str):
    for branch in BRANCHES:
        plot_scatter(
            merged["prior"].to_numpy(),
            merged[f"{branch}_prob"].to_numpy(),
            f"{group_name.title()} prior vs {branch}",
            f"{group_name} prior",
            f"{branch} mean prob",
            os.path.join(output_dir, f"{group_name}_prior_vs_{branch}.png"),
        )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = validate_and_load_pt(args.pt)
    torch.save(data, os.path.join(args.output_dir, "validated_test_predictions.pt"))

    train_csv, test_csv = resolve_csv_paths(args)
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    pred_df=build_prediction_dataframe(data)

    pred_df["factual_prob"] = sigmoid(pred_df["factual_logits"].to_numpy())
    pred_df["cf_drug_prob"] = sigmoid(pred_df["cf_drug_logits"].to_numpy())
    pred_df["cf_protein_prob"] = sigmoid(pred_df["cf_protein_logits"].to_numpy())
    pred_df["debiased_prob"] = sigmoid(pred_df["debiased_logits"].to_numpy())

    alignment_info = validate_test_alignment(pred_df, test_df)
    with open(os.path.join(args.output_dir, "test_alignment_check.json"), "w", encoding="utf-8") as f:
        json.dump(alignment_info, f, ensure_ascii=False, indent=2)

    plot_branch_hist(pred_df, "logits",os.path.join(args.output_dir, "branch_logits_histograms.png"))
    plot_branch_hist(pred_df, "prob",os.path.join(args.output_dir, "branch_probability_histograms.png"))

    pos_neg_stats_df = compute_pos_neg_stats(pred_df)
    pos_neg_stats_df.to_csv(os.path.join(args.output_dir, "branch_pos_neg_stats.csv"), index=False)

    pr_col = _pick_col(train_df, ["pr_id", "target_id", "Target", "Protein"])
    drug_col = _pick_col(train_df, ["SMILES", "Drug", "drug_id"])

    protein_prior = build_group_prior(train_df, pr_col).rename(columns={pr_col: "pr_id"})
    drug_prior = build_group_prior(train_df, drug_col).rename(columns={drug_col: "SMILES"})

    pr_group = pred_df.groupby("pr_id", as_index=False)[[f"{branch}_prob" for branch in BRANCHES]].mean()
    drug_group = pred_df.groupby("SMILES", as_index=False)[[f"{branch}_prob" for branch in BRANCHES]].mean()

    pr_merged = protein_prior.merge(pr_group, on="pr_id", how="inner")
    drug_merged = drug_prior.merge(drug_group, on="SMILES", how="inner")

    pr_corr = summarize_prior_correlations(pr_merged, "protein")
    drug_corr = summarize_prior_correlations(drug_merged, "drug")

    print("[Protein correlation]", json.dumps(pr_corr, ensure_ascii=False, indent=2))
    print("[Drug correlation]", json.dumps(drug_corr, ensure_ascii=False, indent=2))

    with open(os.path.join(args.output_dir, "correlation_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"protein": pr_corr, "drug": drug_corr}, f, ensure_ascii=False, indent=2)

    corr_rows = []
    for scope, corr_dict in (("protein", pr_corr), ("drug", drug_corr)):
        for pair, stats in corr_dict.items():
            if isinstance(stats, dict):
                corr_rows.append({"scope": scope, "pair": pair, **stats})
    pd.DataFrame(corr_rows).to_csv(os.path.join(args.output_dir, "correlation_summary.csv"), index=False)

    pr_merged.to_csv(os.path.join(args.output_dir, "protein_prior_vs_predictions.csv"), index=False)
    drug_merged.to_csv(os.path.join(args.output_dir, "drug_prior_vs_predictions.csv"), index=False)

    plot_group_hist(pr_merged, "protein", os.path.join(args.output_dir, "protein_prior_prediction_hist.png"))
    plot_group_hist(drug_merged, "drug", os.path.join(args.output_dir, "drug_prior_prediction_hist.png"))

    export_scatter_plots(pr_merged, "protein", args.output_dir)
    export_scatter_plots(drug_merged, "drug", args.output_dir)


if __name__ == "__main__":
    main()

