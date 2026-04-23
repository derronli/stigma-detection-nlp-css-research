"""Run comment-level K-fold cross-validation for multitext DCN models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.dcn import DCN
from model.dcnv2 import DCNv2
from trainer_multitext import (
    ANNOTATOR_EMB_DIM,
    DEFAULT_EMB,
    DEFAULT_EMB_ALIGNMENT,
    DEFAULT_FINAL_CSV,
    DEFAULT_MANIFEST,
    DEMO_DIM,
    _pick_device,
    load_arrays,
    maybe_warn_manifest,
)


def k_fold_comment_level_splits(
    comment_ids: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a list of ``(train_row_idx, val_row_idx)`` for K comment-level folds.

    Splits the set of unique ``comment_id`` values into K folds (stratified by
    majority-vote per-comment label when both classes are present), then
    expands back to row indices into ``comment_ids``. The per-comment disjointness
    guarantees the same comment is never in both train and val of a fold.
    """
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold

    cids = np.asarray(comment_ids).astype(str)
    unique_cids = pd.Series(cids).drop_duplicates().to_numpy()

    per_cid_labels = pd.Series(np.asarray(labels, dtype=float))
    per_cid_labels.index = pd.Index(cids, name="cid")
    mean_per_cid = per_cid_labels.groupby(level=0).mean()
    strat = (mean_per_cid.reindex(unique_cids).to_numpy() >= 0.5).astype(int)

    idx_range = np.arange(len(unique_cids))
    if len(np.unique(strat)) > 1:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        raw_splits = list(kf.split(idx_range, strat))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        raw_splits = list(kf.split(idx_range))

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_ix, va_ix in raw_splits:
        val_cid_set = set(unique_cids[va_ix].tolist())
        val_mask = np.fromiter((c in val_cid_set for c in cids), dtype=bool, count=len(cids))
        val_idx = np.flatnonzero(val_mask).astype(np.int64)
        train_idx = np.flatnonzero(~val_mask).astype(np.int64)
        assert set(cids[train_idx].tolist()).isdisjoint(set(cids[val_idx].tolist())), (
            "k_fold_comment_level_splits produced overlapping comment_ids"
        )
        folds.append((train_idx, val_idx))
    return folds


def _binary_metrics_at_threshold(
    y_true: np.ndarray, probs: np.ndarray, threshold: float
) -> dict:
    pred = (probs >= threshold).astype(np.int64)
    y = y_true.astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / max(len(y), 1)

    # AUC is threshold-independent; undefined when only one class is present in y.
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, probs))

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _subset_dataset(
    text: np.ndarray,
    demo: np.ndarray,
    worker: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(text[idx]),
        torch.from_numpy(demo[idx]),
        torch.from_numpy(worker[idx]),
        torch.from_numpy(labels[idx]),
    )


def _evaluate_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (avg_loss, probs, labels) over the full loader."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for text_b, demo_b, worker_b, label_b in loader:
            text_b = text_b.to(device)
            demo_b = demo_b.to(device)
            worker_b = worker_b.to(device)
            label_b = label_b.to(device)
            logits = model(text_b, demo_b, worker_b)
            loss = F.binary_cross_entropy_with_logits(logits, label_b, reduction="sum")
            total_loss += float(loss.item())
            total_n += int(label_b.size(0))
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            labels_all.append(label_b.cpu().numpy())
    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, np.concatenate(probs_all), np.concatenate(labels_all)


def train_one_fold(
    fold_idx: int,
    n_folds: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    text: np.ndarray,
    demo: np.ndarray,
    worker: np.ndarray,
    labels: np.ndarray,
    text_dim: int,
    num_workers_vocab: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train a fresh model on this fold and return (best_val_probs, val_labels, best_val_loss).

    Best epoch is selected by val loss within the fold.
    """
    torch.manual_seed(args.seed + fold_idx)
    np.random.seed(args.seed + fold_idx)

    train_ds = _subset_dataset(text, demo, worker, labels, train_idx)
    val_ds = _subset_dataset(text, demo, worker, labels, val_idx)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin
    )

    input_dim = text_dim + DEMO_DIM + ANNOTATOR_EMB_DIM
    model_cls = DCN if args.model == "dcn" else DCNv2
    model = model_cls(
        input_dim=input_dim,
        num_workers=num_workers_vocab,
        annotator_emb_dim=ANNOTATOR_EMB_DIM,
        text_dim=text_dim,
        demo_dim=DEMO_DIM,
        num_cross_layers=args.num_cross_layers,
        num_deep_layers=args.num_deep_layers,
        deep_layer_size=args.deep_layer_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_probs: np.ndarray | None = None
    best_labels: np.ndarray | None = None

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        pbar_desc = f"[Fold {fold_idx + 1}/{n_folds}] Epoch {epoch + 1}/{args.epochs} [train]"
        train_pbar = tqdm(train_loader, desc=pbar_desc, leave=False, dynamic_ncols=True)
        for text_emb, demo_b, worker_b, label in train_pbar:
            text_emb = text_emb.to(device)
            demo_b = demo_b.to(device)
            worker_b = worker_b.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(text_emb, demo_b, worker_b)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            running_loss += loss.item() * label.size(0)
            running_correct += (preds == label).sum().item()
            running_count += label.size(0)
            train_pbar.set_postfix(
                loss=f"{running_loss / max(running_count, 1):.4f}",
                acc=f"{running_correct / max(running_count, 1):.4f}",
            )

        train_loss = running_loss / max(running_count, 1)

        val_loss, val_probs, val_labels = _evaluate_probs(model, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"[Fold {fold_idx + 1}/{n_folds}] "
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_probs = val_probs
            best_labels = val_labels

    assert best_probs is not None and best_labels is not None, (
        "No epoch completed for this fold; check --epochs"
    )
    return best_probs, best_labels, best_val_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-fold CV for multitext DCN / DCNv2")
    p.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMB)
    p.add_argument(
        "--embedding-alignment",
        "--comment-ids",
        type=Path,
        default=DEFAULT_EMB_ALIGNMENT,
        dest="embedding_alignment",
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--included-indices", type=Path, default=None)

    p.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Decision threshold used to compute per-fold F1.",
    )

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-cross-layers", type=int, default=3)
    p.add_argument("--num-deep-layers", type=int, default=2)
    p.add_argument("--deep-layer-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--model",
        choices=["dcn", "dcnv2"],
        default="dcn",
        help="Model architecture to train.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_splits < 2:
        raise ValueError(f"--n-splits must be >= 2, got {args.n_splits}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError(f"--threshold must be in (0, 1), got {args.threshold}")

    text, demo, worker, labels, worker_map, text_dim, comment_ids = load_arrays(
        args.final_csv,
        args.embeddings,
        args.embedding_alignment,
        args.included_indices,
    )
    maybe_warn_manifest(args.manifest, text_dim)

    device = _pick_device()
    folds = k_fold_comment_level_splits(comment_ids, labels, args.n_splits, args.seed)

    print(
        f"Running {args.n_splits}-fold CV  |  {len(labels)} rows, "
        f"{len(np.unique(comment_ids))} unique comment_ids  |  device={device}  "
        f"|  threshold={args.threshold}"
    )

    fold_metrics: list[dict] = []
    for i, (train_idx, val_idx) in enumerate(folds):
        print(
            f"\n=== Fold {i + 1}/{args.n_splits} ===  "
            f"train rows={len(train_idx)}, val rows={len(val_idx)}"
        )
        best_probs, val_labels, best_val_loss = train_one_fold(
            fold_idx=i,
            n_folds=args.n_splits,
            train_idx=train_idx,
            val_idx=val_idx,
            text=text,
            demo=demo,
            worker=worker,
            labels=labels,
            text_dim=text_dim,
            num_workers_vocab=len(worker_map),
            args=args,
            device=device,
        )
        m = _binary_metrics_at_threshold(val_labels, best_probs, args.threshold)
        m["best_val_loss"] = best_val_loss
        fold_metrics.append(m)
        print(
            f"[Fold {i + 1}/{args.n_splits}] F1@{args.threshold:.2f} = {m['f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"acc={m['accuracy']:.4f}  AUC={m['auc']:.4f}  "
            f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}  "
            f"(best val_loss={best_val_loss:.4f})"
        )

    f1s = np.array([m["f1"] for m in fold_metrics], dtype=np.float64)
    precisions = np.array([m["precision"] for m in fold_metrics], dtype=np.float64)
    recalls = np.array([m["recall"] for m in fold_metrics], dtype=np.float64)
    accuracies = np.array([m["accuracy"] for m in fold_metrics], dtype=np.float64)
    aucs = np.array([m["auc"] for m in fold_metrics], dtype=np.float64)

    print(f"\n=== {args.n_splits}-Fold CV Summary  (threshold = {args.threshold}) ===")
    for i, m in enumerate(fold_metrics):
        print(
            f"  Fold {i + 1}:  F1={m['f1']:.4f}  "
            f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"acc={m['accuracy']:.4f}  AUC={m['auc']:.4f}"
        )
    print(
        f"  ---------------------------------------------------"
    )
    print(
        f"  Mean : F1={f1s.mean():.4f}  "
        f"P={precisions.mean():.4f}  R={recalls.mean():.4f}  "
        f"acc={accuracies.mean():.4f}  AUC={np.nanmean(aucs):.4f}"
    )
    print(
        f"  Std  : F1={f1s.std(ddof=0):.4f}  "
        f"P={precisions.std(ddof=0):.4f}  R={recalls.std(ddof=0):.4f}  "
        f"acc={accuracies.std(ddof=0):.4f}  AUC={np.nanstd(aucs, ddof=0):.4f}"
    )


if __name__ == "__main__":
    main()
