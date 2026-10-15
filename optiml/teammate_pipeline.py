"""Portable two-view CatBoost pipeline for an NVIDIA RTX 3050.

The pipeline trains two deliberately different models:

* ``line_view`` uses transaction, product, customer, and store fields.
* ``order_view`` uses target-independent basket aggregates plus the line view.

Order aggregates are calculated separately inside each temporal partition. They
use no labels, so the complete order is available at prediction time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .features import build_features, dimensions
from .prepare import sha256
from .train import write_submission


def _backend(device: str) -> dict[str, str]:
    if device == "CPU":
        return {"task_type": "CPU"}
    if device == "GPU":
        return {"task_type": "GPU", "devices": "0"}
    raise ValueError("device must be CPU or GPU; GPU never silently falls back to CPU")


def add_order_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add input-only basket aggregates without changing row order."""
    result = frame.copy()
    numeric = [
        c for c in (
            "unit_price", "quantity", "discount", "freight_cost", "order_value",
            "session_clicks", "line_no", "product_weight_kg", "return_window_days",
        )
        if c in result.columns and pd.api.types.is_numeric_dtype(result[c])
    ]
    grouped = result.groupby("order_id", sort=False, dropna=False)
    result["ctx_line_count"] = grouped["order_line_id"].transform("size").astype("float32")
    for column in numeric:
        values = grouped[column].transform("mean")
        result[f"ctx_{column}_mean"] = values.astype("float32")
        result[f"ctx_{column}_delta"] = (
            result[column].astype("float32") - values.astype("float32")
        )
    if "product_id" in result:
        result["ctx_distinct_products"] = grouped["product_id"].transform("nunique").astype("float32")
    if "store_id" in result:
        result["ctx_distinct_stores"] = grouped["store_id"].transform("nunique").astype("float32")
    if "product_category" in result:
        result["ctx_distinct_categories"] = grouped["product_category"].transform("nunique").astype("float32")
    return result


def _partition(frame: pd.DataFrame, cutoff: str, horizon_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_ts = pd.Timestamp(cutoff)
    fit = frame.loc[frame["_order_time"] < cutoff_ts].copy()
    valid = frame.loc[
        (frame["_order_time"] >= cutoff_ts)
        & (frame["_order_time"] < cutoff_ts + pd.DateOffset(months=horizon_months))
    ].copy()
    if fit.empty or valid.empty or fit.returned.nunique() != 2 or valid.returned.nunique() != 2:
        raise ValueError("Temporal partition must contain both target classes")
    if set(fit.order_id).intersection(valid.order_id):
        raise AssertionError("An order crosses the temporal boundary")
    return fit, valid


def _columns(frame: pd.DataFrame, view: str) -> list[str]:
    excluded = {"returned", "order_line_id", "order_id", "_order_time", "_month"}
    if view == "line_view":
        excluded |= {c for c in frame if c.startswith("ctx_")}
    else:
        excluded |= {"customer_id", "product_id", "store_id"}
    columns = [c for c in frame if c not in excluded]
    return [c for c in columns if not pd.api.types.is_datetime64_any_dtype(frame[c])]


def _prepare_catboost(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    data = frame[columns].copy()
    categorical: list[str] = []
    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            categorical.append(column)
            data[column] = data[column].astype("string").fillna("__MISSING__").astype(str)
        else:
            data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    return data, categorical


def _fit_predict(
    fit: pd.DataFrame,
    valid: pd.DataFrame,
    view: str,
    args: argparse.Namespace,
    output: Path,
) -> tuple[np.ndarray, int]:
    from catboost import CatBoostClassifier

    columns = _columns(fit, view)
    x_fit, categories = _prepare_catboost(fit, columns)
    x_valid, _ = _prepare_catboost(valid, columns)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=6.0,
        random_seed=args.seed,
        max_ctr_complexity=1,
        thread_count=args.threads,
        task_type=_backend(args.device)["task_type"],
        devices="0" if args.device == "GPU" else None,
        od_type="Iter",
        od_wait=args.patience,
        allow_writing_files=False,
    )
    model.fit(
        x_fit,
        fit.returned,
        cat_features=categories,
        eval_set=(x_valid, valid.returned),
        verbose=False,
    )
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output / f"{view}.cbm"))
    (output / f"{view}_features.json").write_text(json.dumps(columns, indent=2))
    best = max(1, model.get_best_iteration() + 1)
    return model.predict_proba(x_valid)[:, 1], best


def _auc(y: pd.Series, prediction: np.ndarray) -> float:
    return float(roc_auc_score(y, prediction))


def _blend(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    # AUC is rank based; averaging logits avoids over-weighting saturated scores.
    def logit(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x))

    logits = (1.0 - weight) * logit(a) + weight * logit(b)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def _load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ("train", "test", "sample_submission")
    missing = [name for name in required if not (data_dir / f"{name}.csv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(name + '.csv' for name in missing)} in {data_dir}. "
            "Obtain the authorized competition data and do not download substitutes."
        )
    dims = dimensions(data_dir)
    train = build_features(pd.read_csv(data_dir / "train.csv", low_memory=False), dims)
    test = build_features(pd.read_csv(data_dir / "test.csv", low_memory=False), dims)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if list(sample.columns) != ["order_line_id", "returned"]:
        raise ValueError("sample_submission.csv must contain order_line_id and returned")
    return train, test, sample


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, test, sample = _load_data(args.data_dir)
    train = add_order_context(train)
    test = add_order_context(test)

    fit, valid = _partition(train, args.cutoff, args.horizon_months)
    predictions: dict[str, np.ndarray] = {}
    best_iterations: dict[str, int] = {}
    for view in ("line_view", "order_view"):
        predictions[view], best_iterations[view] = _fit_predict(
            fit, valid, view, args, args.output_dir / "models"
        )
    candidates = {
        weight: _auc(valid.returned, _blend(predictions["line_view"], predictions["order_view"], weight))
        for weight in (0.25, 0.50, 0.75)
    }
    weight = max(candidates, key=candidates.get)
    assessment_prediction = _blend(predictions["line_view"], predictions["order_view"], weight)
    metrics = {
        "pipeline": "two-view-order-context-v1",
        "cutoff": args.cutoff,
        "horizon_months": args.horizon_months,
        "fit_rows": len(fit),
        "validation_rows": len(valid),
        "line_auc": _auc(valid.returned, predictions["line_view"]),
        "order_auc": _auc(valid.returned, predictions["order_view"]),
        "blend_candidates": {str(k): v for k, v in candidates.items()},
        "selected_order_view_weight": weight,
        "blend_auc": _auc(valid.returned, assessment_prediction),
        "blend_log_loss": float(log_loss(valid.returned, assessment_prediction)),
        "best_iterations": best_iterations,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({
        "order_line_id": valid.order_line_id,
        "returned": valid.returned,
        "line_prediction": predictions["line_view"],
        "order_prediction": predictions["order_view"],
        "prediction": assessment_prediction,
    }).to_parquet(args.output_dir / "validation_predictions.parquet", index=False)

    if args.finalize:
        final_predictions: dict[str, np.ndarray] = {}
        for view in ("line_view", "order_view"):
            final_args = argparse.Namespace(**vars(args))
            final_args.iterations = best_iterations[view]
            final_predictions[view], _ = _fit_predict(
                train, test, view, final_args, args.output_dir / "models"
            )
        final = _blend(
            final_predictions["line_view"], final_predictions["order_view"], weight
        )
        write_submission(sample, test.order_line_id, final, args.output_dir / "submission.csv")
        (args.output_dir / "FINALIZED.json").write_text(json.dumps({
            "submission_rows": len(final),
            "data_sha256": {
                name: sha256(args.data_dir / f"{name}.csv")
                for name in ("train", "test", "sample_submission")
            },
        }, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/teammate-two-view"))
    parser.add_argument("--cutoff", default="2024-07-01")
    parser.add_argument("--horizon-months", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
