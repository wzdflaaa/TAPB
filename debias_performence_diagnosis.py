import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

REQUIRED_KEYS = {
    "label",
    "factual_logits",
    "cf_drug_logits",
    "cf_protein_logits",
    "debiased_logits",
    "pr_id",
    "SMILES",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2: debias performance diagnosis")
    parser.add_argument("--pt", required=True, type=str, help="path to test_predictions.pt")
    parser.add_argument("--train_csv", required=True, type=str, help="path to train csv")
    parser.add_argument("--test_csv", required=True, type=str, help="path to test csv")
    parser.add_argument("--output_dir", required=True, type=str, help="output directory")
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def _main_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    auroc, auprc = _safe_auc(y_true, y_prob)
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": auroc,
            "auprc": auprc,
            "f1": float("nan"),
            "accuracy": float("nan"),
            "threshold": float("nan"),
            "mcc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = int(np.argmax(tpr - fpr))
    th = float(thresholds[idx])
    y_bin = (y_prob >= th).astype(int)

    cm = confusion_matrix(y_true, y_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    mcc = float(matthews_corrcoef(y_true, y_bin)) if len(np.unique(y_bin)) > 1 else float("nan")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_true, y_bin)),
        "accuracy": float(accuracy_score(y_true, y_bin)),
        "threshold": th,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _pick_col(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of columns found: {candidates}")


def _build_prior(train_df: pd.DataFrame, group_col: str, label_col: str = "Y") -> pd.DataFrame:
    tmp = train_df[[group_col, label_col]].copy()
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce").fillna(0).astype(int)
    tmp[label_col] = (tmp[label_col] > 0).astype(int)
    return tmp.groupby(group_col, as_index=False)[label_col].mean().rename(columns={label_col: "prior"})


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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


def _bias_reduction_ratio(c_factual: float, c_debiased: float) -> float:
    if np.isnan(c_factual) or c_factual == 0:
        return float("nan")
    return float((abs(c_factual) - abs(c_debiased)) / abs(c_factual))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = validate_and_load_pt(args.pt)
    torch.save(data, os.path.join(args.output_dir, "validated_test_predictions.pt"))

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    pred_df = pd.DataFrame(
        {
            "label": np.asarray(data["label"], dtype=int),
            "pr_id": data["pr_id"],
            "SMILES": data["SMILES"],
            "factual": sigmoid(np.asarray(data["factual_logits"], dtype=float)),
            "cf_drug": sigmoid(np.asarray(data["cf_drug_logits"], dtype=float)),
            "cf_protein": sigmoid(np.asarray(data["cf_protein_logits"], dtype=float)),
            "debiased": sigmoid(np.asarray(data["debiased_logits"], dtype=float)),
        }
    )

    if len(pred_df) != len(test_df):
        raise ValueError(f"Length mismatch between pred_df ({len(pred_df)}) and test_csv ({len(test_df)})")

    y = pred_df["label"].to_numpy()
    branches = ["factual", "cf_drug", "cf_protein", "debiased"]

    main_metrics = []
    for b in branches:
        m = _main_metrics(y, pred_df[b].to_numpy())
        m["branch"] = b
        main_metrics.append(m)

    main_metrics_df = pd.DataFrame(main_metrics)[
        ["branch", "auroc", "auprc", "f1", "accuracy", "threshold", "mcc", "sensitivity", "specificity"]
    ]
    main_metrics_df.to_csv(os.path.join(args.output_dir, "main_branch_metrics.csv"), index=False)

    factual_auroc = float(main_metrics_df.loc[main_metrics_df["branch"] == "factual", "auroc"].iloc[0])
    cf_drug_auroc = float(main_metrics_df.loc[main_metrics_df["branch"] == "cf_drug", "auroc"].iloc[0])
    cf_protein_auroc = float(main_metrics_df.loc[main_metrics_df["branch"] == "cf_protein", "auroc"].iloc[0])

    factual_auprc = float(main_metrics_df.loc[main_metrics_df["branch"] == "factual", "auprc"].iloc[0])
    cf_drug_auprc = float(main_metrics_df.loc[main_metrics_df["branch"] == "cf_drug", "auprc"].iloc[0])
    cf_protein_auprc = float(main_metrics_df.loc[main_metrics_df["branch"] == "cf_protein", "auprc"].iloc[0])

    counterfactual_gap_df = pd.DataFrame(
        [
            {
                "factual_auroc": factual_auroc,
                "cf_drug_auroc": cf_drug_auroc,
                "cf_protein_auroc": cf_protein_auroc,
                "factual_minus_cf_drug_auroc": factual_auroc - cf_drug_auroc,
                "factual_minus_cf_protein_auroc": factual_auroc - cf_protein_auroc,
                "AUROC_drop_cf_drug_proxy": factual_auroc - cf_drug_auroc,
                "AUROC_drop_cf_protein_proxy": factual_auroc - cf_protein_auroc,
                "AUPRC_drop_cf_drug_proxy": factual_auprc - cf_drug_auprc,
                "AUPRC_drop_cf_protein_proxy": factual_auprc - cf_protein_auprc,
            }
        ]
    )
    counterfactual_gap_df.to_csv(os.path.join(args.output_dir, "counterfactual_auroc_gaps.csv"), index=False)

    pr_col = _pick_col(train_df, ["pr_id", "target_id", "Target", "Protein"])
    drug_col = _pick_col(train_df, ["SMILES", "Drug", "drug_id"])

    protein_prior = _build_prior(train_df, pr_col).rename(columns={pr_col: "pr_id"})
    drug_prior = _build_prior(train_df, drug_col).rename(columns={drug_col: "SMILES"})

    pr_group = pred_df.groupby("pr_id", as_index=False)[["factual", "cf_protein", "debiased"]].mean()
    drug_group = pred_df.groupby("SMILES", as_index=False)[["factual", "cf_drug", "debiased"]].mean()

    pr_merged = protein_prior.merge(pr_group, on="pr_id", how="inner")
    drug_merged = drug_prior.merge(drug_group, on="SMILES", how="inner")

    corr_prior_factual_protein = _corr(pr_merged["prior"].to_numpy(), pr_merged["factual"].to_numpy())
    corr_prior_cf_protein = _corr(pr_merged["prior"].to_numpy(), pr_merged["cf_protein"].to_numpy())
    corr_prior_debiased_protein = _corr(pr_merged["prior"].to_numpy(), pr_merged["debiased"].to_numpy())

    corr_prior_factual_drug = _corr(drug_merged["prior"].to_numpy(), drug_merged["factual"].to_numpy())
    corr_prior_cf_drug = _corr(drug_merged["prior"].to_numpy(), drug_merged["cf_drug"].to_numpy())
    corr_prior_debiased_drug = _corr(drug_merged["prior"].to_numpy(), drug_merged["debiased"].to_numpy())

    corr_summary = {
        "protein": {
            "corr_prior_factual": corr_prior_factual_protein,
            "corr_prior_counterfactual": corr_prior_cf_protein,
            "corr_prior_debiased": corr_prior_debiased_protein,
            "BRR_target": _bias_reduction_ratio(corr_prior_factual_protein, corr_prior_debiased_protein),
            "n_overlap_groups": int(len(pr_merged)),
        },
        "drug": {
            "corr_prior_factual": corr_prior_factual_drug,
            "corr_prior_counterfactual": corr_prior_cf_drug,
            "corr_prior_debiased": corr_prior_debiased_drug,
            "BRR_drug": _bias_reduction_ratio(corr_prior_factual_drug, corr_prior_debiased_drug),
            "n_overlap_groups": int(len(drug_merged)),
        },
    }

    pd.DataFrame(
        [
            {"scope": "protein", **corr_summary["protein"]},
            {"scope": "drug", **corr_summary["drug"]},
        ]
    ).to_csv(os.path.join(args.output_dir, "prior_prediction_correlations.csv"), index=False)

    with open(os.path.join(args.output_dir, "diagnosis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "main_branch_metrics": main_metrics,
                "counterfactual_gap": counterfactual_gap_df.to_dict(orient="records"),
                "correlation_summary": corr_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(corr_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()