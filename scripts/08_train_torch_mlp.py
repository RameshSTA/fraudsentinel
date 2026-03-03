from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.config import ProjectPaths
from src.modeling.torch_mlp import (
    TabularFraudDataset,
    TorchConfig,
    TorchMLP,
    build_feature_schema,
    fit_standardizer,
    save_json,
    transform_keys,
    transform_numeric,
)
from src.utils.logging import setup_logger


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "avg_precision": float(average_precision_score(y_true, y_score)),
    }


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Predict probabilities for a DataLoader.

    Supports loaders yielding (x_num, x_ent, x_fp) or (x_num, x_ent, x_fp, y).
    """
    model.eval()
    scores: list[np.ndarray] = []

    for batch in loader:
        if len(batch) == 4:
            x_num, x_ent, x_fp, _y = batch
        else:
            x_num, x_ent, x_fp = batch

        logits = model(x_num.to(device), x_ent.to(device), x_fp.to(device))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        scores.append(probs)

    return np.concatenate(scores, axis=0)


def main() -> None:
    """
    Step 7B: Train PyTorch MLP (embeddings + numeric) with MLflow tracking.

    Input:
        data/processed/dataset_v1.parquet

    Outputs:
        models/torch_mlp.pt
        models/torch_feature_schema.json
        reports/torch_mlp_metrics.json
    """
    logger = setup_logger()
    paths = ProjectPaths()

    cfg = TorchConfig()

    in_path = paths.data_processed / "dataset_v1.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run Step 6 first.")

    logger.info("Loading dataset: %s", in_path)
    df = pd.read_parquet(in_path)

    required = {cfg.target_col, cfg.split_col, "entity_key", "fingerprint_key"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    schema = build_feature_schema(df)
    numeric_cols = schema["numeric_cols"]

    train_df = df[df[cfg.split_col] == "train"].copy()
    valid_df = df[df[cfg.split_col] == "valid"].copy()
    test_df = df[df[cfg.split_col] == "test"].copy()

    logger.info("Split sizes: train=%d valid=%d test=%d", len(train_df), len(valid_df), len(test_df))

    # Fit standardizer on train only (training/serving parity guaranteed)
    logger.info("Fitting numeric standardizer on train split only")
    stats = fit_standardizer(train_df, numeric_cols)

    Xtr = transform_numeric(train_df, numeric_cols, stats)
    Xva = transform_numeric(valid_df, numeric_cols, stats)
    Xte = transform_numeric(test_df, numeric_cols, stats)

    ent_tr = transform_keys(train_df, "entity_key", cfg.entity_buckets)
    ent_va = transform_keys(valid_df, "entity_key", cfg.entity_buckets)
    ent_te = transform_keys(test_df, "entity_key", cfg.entity_buckets)

    fp_tr = transform_keys(train_df, "fingerprint_key", cfg.fingerprint_buckets)
    fp_va = transform_keys(valid_df, "fingerprint_key", cfg.fingerprint_buckets)
    fp_te = transform_keys(test_df, "fingerprint_key", cfg.fingerprint_buckets)

    ytr = train_df[cfg.target_col].astype(int).to_numpy()
    yva = valid_df[cfg.target_col].astype(int).to_numpy()
    yte = test_df[cfg.target_col].astype(int).to_numpy()

    # Class imbalance: pos_weight = neg / pos
    pos = float(ytr.sum())
    neg = float(len(ytr) - ytr.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s", device)

    train_ds = TabularFraudDataset(Xtr, ent_tr, fp_tr, ytr.astype(np.float32))
    valid_ds = TabularFraudDataset(Xva, ent_va, fp_va, yva.astype(np.float32))
    test_ds = TabularFraudDataset(Xte, ent_te, fp_te, yte.astype(np.float32))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = TorchMLP(
        num_dim=Xtr.shape[1],
        entity_buckets=cfg.entity_buckets,
        fingerprint_buckets=cfg.fingerprint_buckets,
        entity_emb_dim=cfg.entity_emb_dim,
        fingerprint_emb_dim=cfg.fingerprint_emb_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # ─── MLflow Experiment Tracking ───────────────────────────────────────────
    mlflow.set_experiment("scam-detection-torch-mlp")

    with mlflow.start_run(run_name="torch-mlp-embeddings-v1"):
        mlflow.log_params({
            "model": "TorchMLP",
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "entity_emb_dim": cfg.entity_emb_dim,
            "fingerprint_emb_dim": cfg.fingerprint_emb_dim,
            "entity_buckets": cfg.entity_buckets,
            "fingerprint_buckets": cfg.fingerprint_buckets,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "max_epochs": cfg.max_epochs,
            "patience": cfg.patience,
            "device": str(device),
            "label_horizon_h": 72,
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "test_rows": len(test_df),
            "pos_weight": round(neg / max(pos, 1.0), 2),
        })

        best_valid_ap = -1.0
        best_state = None
        bad_epochs = 0

        logger.info(
            "Training started: max_epochs=%d batch_size=%d device=%s",
            cfg.max_epochs,
            cfg.batch_size,
            device,
        )

        for epoch in range(1, cfg.max_epochs + 1):
            t0 = time.time()
            model.train()
            running_loss = 0.0
            n = 0

            for x_num, x_ent, x_fp, y in train_loader:
                x_num, x_ent, x_fp, y = (
                    x_num.to(device),
                    x_ent.to(device),
                    x_fp.to(device),
                    y.to(device),
                )
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x_num, x_ent, x_fp), y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * len(y)
                n += len(y)

            train_loss = running_loss / max(n, 1)
            valid_scores = _predict(model, valid_loader, device)
            valid_metrics = _metrics(yva, valid_scores)

            # Log per-epoch metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "valid_auc_roc": valid_metrics["auc_roc"],
                    "valid_avg_precision": valid_metrics["avg_precision"],
                },
                step=epoch,
            )

            dt = time.time() - t0
            logger.info(
                "Epoch %d | train_loss=%.6f | valid_auc=%.4f valid_ap=%.4f | %.1fs",
                epoch,
                train_loss,
                valid_metrics["auc_roc"],
                valid_metrics["avg_precision"],
                dt,
            )

            if valid_metrics["avg_precision"] > best_valid_ap + 1e-5:
                best_valid_ap = valid_metrics["avg_precision"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch, cfg.patience)
                    break

        if best_state is None:
            raise RuntimeError("Training failed to produce a best model state.")

        model.load_state_dict(best_state)

        # Final evaluation on best checkpoint
        valid_scores = _predict(model, valid_loader, device)
        test_scores = _predict(model, test_loader, device)
        final_valid = _metrics(yva, valid_scores)
        final_test = _metrics(yte, test_scores)

        mlflow.log_metrics({
            "final_valid_auc_roc": final_valid["auc_roc"],
            "final_valid_avg_precision": final_valid["avg_precision"],
            "final_test_auc_roc": final_test["auc_roc"],
            "final_test_avg_precision": final_test["avg_precision"],
        })

        # ─── Save artefacts ───────────────────────────────────────────────────
        model_path = Path("models/torch_mlp.pt")
        schema_path = Path("models/torch_feature_schema.json")
        report_path = paths.reports / "torch_mlp_metrics.json"

        model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "numeric_cols": numeric_cols,
                "scaler_stats": stats,
                "entity_buckets": cfg.entity_buckets,
                "fingerprint_buckets": cfg.fingerprint_buckets,
            },
            model_path,
        )

        save_json(
            {
                "numeric_cols": numeric_cols,
                "scaler_stats": stats,
                "entity_buckets": cfg.entity_buckets,
                "fingerprint_buckets": cfg.fingerprint_buckets,
            },
            schema_path,
        )

        report = {
            "model": "torch_mlp_embeddings",
            "device": str(device),
            "config": cfg.__dict__,
            "valid_metrics": final_valid,
            "test_metrics": final_test,
            "notes": (
                "Hashed entity_key & fingerprint_key embeddings + "
                "standardized numeric features + MLP head."
            ),
        }
        save_json(report, report_path)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(schema_path))
        mlflow.log_artifact(str(report_path))

        logger.info("Saved model: %s", model_path)
        logger.info("Saved schema: %s", schema_path)
        logger.info("Saved report: %s", report_path)
        logger.info(
            "Done. valid_auc=%.4f valid_ap=%.4f test_auc=%.4f test_ap=%.4f",
            final_valid["auc_roc"],
            final_valid["avg_precision"],
            final_test["auc_roc"],
            final_test["avg_precision"],
        )
        logger.info("MLflow run logged. Run 'make mlflow-ui' to explore.")


if __name__ == "__main__":
    main()
