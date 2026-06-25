#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def _load_face_spectrum(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON payload in {path}")
    arr = np.asarray(data.get("intensity", []), dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        raise ValueError(f"Empty intensity in {path}")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"Non-finite intensity values in {path}")
    return arr


def load_dataset(data_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    samples: list[np.ndarray] = []
    labels: list[float] = []
    paths: list[str] = []
    for cls_name, label in (("real", 1.0), ("fake", 0.0)):
        cls_root = data_root / cls_name
        if not cls_root.exists():
            continue
        json_paths = sorted(cls_root.glob("**/face_spectrum.json"))
        for js in json_paths:
            vec = _load_face_spectrum(js)
            samples.append(vec)
            labels.append(label)
            paths.append(str(js))
    if not samples:
        raise RuntimeError(f"No face_spectrum.json found in {data_root}")
    feat_dim = int(samples[0].size)
    for i, vec in enumerate(samples):
        if int(vec.size) != feat_dim:
            raise RuntimeError(f"Inconsistent feature count at {paths[i]}: {int(vec.size)} vs {feat_dim}")
    x = np.vstack(samples).astype(np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return x, y, paths


def fit_logreg_l2(x: np.ndarray, y: np.ndarray, reg_lambda: float, max_iter: int = 2000) -> dict[str, np.ndarray | float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = x.shape
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    z = (x - mu) / sigma

    def objective(theta: np.ndarray) -> float:
        w = theta[:-1]
        b = float(theta[-1])
        s = z @ w + b
        loss = np.logaddexp(0.0, s) - y * s
        return float(loss.mean() + 0.5 * reg_lambda * np.dot(w, w))

    def gradient(theta: np.ndarray) -> np.ndarray:
        w = theta[:-1]
        b = float(theta[-1])
        s = z @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(s, -60.0, 60.0)))
        e = p - y
        gw = (z.T @ e) / float(n) + reg_lambda * w
        gb = float(e.mean())
        return np.concatenate([gw, np.asarray([gb], dtype=np.float64)])

    theta0 = np.zeros(d + 1, dtype=np.float64)
    res = minimize(
        objective,
        theta0,
        method="L-BFGS-B",
        jac=gradient,
        options={"maxiter": int(max_iter)},
    )
    if not bool(res.success):
        raise RuntimeError(f"Optimizer failed: {res.message}")
    w = np.asarray(res.x[:-1], dtype=np.float64)
    b = float(res.x[-1])
    return {
        "mean": mu,
        "std": sigma,
        "weights": w,
        "bias": b,
    }


def predict_live_probability(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(model["std"], dtype=np.float64).reshape(-1)
    weights = np.asarray(model["weights"], dtype=np.float64).reshape(-1)
    bias = float(model["bias"])
    xx = np.asarray(x, dtype=np.float64)
    z = (xx - mean[None, :]) / std[None, :]
    scores = z @ weights + bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -60.0, 60.0)))
    return probs.astype(np.float64)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    pos = y_true == 1
    neg = y_true == 0
    tpr = float((y_pred[pos] == 1).sum()) / float(max(1, int(pos.sum())))
    tnr = float((y_pred[neg] == 0).sum()) / float(max(1, int(neg.sum())))
    return 0.5 * (tpr + tnr)


def loo_score_for_lambda(x: np.ndarray, y: np.ndarray, reg_lambda: float) -> dict[str, float]:
    n = int(x.shape[0])
    probs = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model_i = fit_logreg_l2(x[mask], y[mask], reg_lambda=reg_lambda)
        probs[i] = float(predict_live_probability(model_i, x[i : i + 1])[0])
    pred = (probs >= 0.5).astype(np.int32)
    y_i = y.astype(np.int32)
    acc = float((pred == y_i).mean())
    bacc = float(balanced_accuracy(y_i, pred))
    return {"accuracy": acc, "balanced_accuracy": bacc}


def train_and_save(data_root: Path, output_model: Path) -> None:
    x, y, _paths = load_dataset(data_root)
    real_n = int((y == 1.0).sum())
    fake_n = int((y == 0.0).sum())
    if real_n < 2 or fake_n < 2:
        raise RuntimeError(f"Need at least 2 samples per class, got real={real_n}, fake={fake_n}")

    reg_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    best_lambda = None
    best_bacc = -1.0
    cv_rows: list[dict[str, float]] = []
    for lam in reg_grid:
        score = loo_score_for_lambda(x, y, reg_lambda=float(lam))
        row = {"lambda": float(lam), **score}
        cv_rows.append(row)
        if row["balanced_accuracy"] > best_bacc:
            best_bacc = row["balanced_accuracy"]
            best_lambda = float(lam)
    if best_lambda is None:
        raise RuntimeError("Failed to select regularization value")

    model = fit_logreg_l2(x, y, reg_lambda=best_lambda)
    probs_train = predict_live_probability(model, x)
    pred_train = (probs_train >= 0.5).astype(np.int32)
    y_i = y.astype(np.int32)
    train_acc = float((pred_train == y_i).mean())
    train_bacc = float(balanced_accuracy(y_i, pred_train))

    payload = {
        "version": 1,
        "model_type": "logreg_l2_numpy",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature_count": int(x.shape[1]),
        "class_names": {"0": "Fake Face", "1": "Live Face"},
        "threshold_live": 0.5,
        "reg_lambda": float(best_lambda),
        "cv_method": "leave_one_out",
        "cv_scores": cv_rows,
        "train_metrics": {
            "accuracy": train_acc,
            "balanced_accuracy": train_bacc,
            "samples_total": int(x.shape[0]),
            "samples_real": real_n,
            "samples_fake": fake_n,
        },
        "mean": [float(v) for v in np.asarray(model["mean"], dtype=np.float64).tolist()],
        "std": [float(v) for v in np.asarray(model["std"], dtype=np.float64).tolist()],
        "weights": [float(v) for v in np.asarray(model["weights"], dtype=np.float64).tolist()],
        "bias": float(model["bias"]),
    }
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_model.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model: {output_model}")
    print(f"Samples: total={int(x.shape[0])}, real={real_n}, fake={fake_n}")
    print(f"Best lambda: {best_lambda}")
    print(f"LOO balanced accuracy: {best_bacc:.4f}")
    print(f"Train accuracy: {train_acc:.4f}, train balanced accuracy: {train_bacc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train face-spectrum real/fake classifier")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cls_data",
        help="Path to cls_data directory with real/fake subfolders",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "weights" / "face_cls_model.json",
        help="Output model JSON path",
    )
    args = parser.parse_args()
    train_and_save(data_root=args.data_root, output_model=args.output_model)


if __name__ == "__main__":
    main()
