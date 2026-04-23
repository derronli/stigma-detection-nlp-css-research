"""Train DCN/DCNv2 on single-text BERTweet embeddings."""

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

from data_split import comment_level_split_indices
from model.dcn import DCN
from model.dcnv2 import DCNv2

PROJECT_ROOT = _SRC.parent
DEFAULT_FINAL_CSV = PROJECT_ROOT / "Data/Dan annotations5k 04202022/final.csv"
DEFAULT_EMB = PROJECT_ROOT / "Data/embeddings/embeddings.npy"
DEFAULT_EMB_ALIGNMENT = PROJECT_ROOT / "Data/embeddings/comment_ids.npy"

DEMO_COLS = ["female", "knowsud", "black", "usesubstance"]
LABEL_COL = "stigma_value"
WORKER_COL = "worker_id"
ID_COL = "comment_id"

TEXT_EMB_DIM = 768
DEMO_DIM = len(DEMO_COLS)
ANNOTATOR_EMB_DIM = 8
INPUT_DIM = TEXT_EMB_DIM + DEMO_DIM + ANNOTATOR_EMB_DIM


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_embedding_alignment_path(path: Path) -> Path:
    """Prefer embedding_alignment.npy; fall back to legacy comment_ids.npy if present."""
    if path.exists():
        return path
    legacy = path.parent / "comment_ids.npy"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"No embedding alignment file at {path} or {legacy}. "
        "Run src/bert.py or pass --embedding-alignment PATH."
    )


def load_arrays(
    final_csv: Path,
    embeddings_path: Path,
    embedding_alignment_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    """Return text_embeds [N,768], demo [N,4], worker_idx [N], labels [N], worker_to_idx, comment_ids [N].

    embedding_alignment_path: one key per row in embeddings.npy (e.g. comment_id string).
    Used only to index into embeddings — never fed to DCN. ``comment_ids`` is
    returned so the caller can produce a comment-level (leakage-free) train/val
    split; it is not a model input.
    """
    import pandas as pd

    alignment_file = resolve_embedding_alignment_path(embedding_alignment_path)
    df = pd.read_csv(final_csv)
    for c in [WORKER_COL, ID_COL, LABEL_COL, *DEMO_COLS]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {final_csv}")

    emb = np.load(embeddings_path)
    align_keys = np.load(alignment_file, allow_pickle=True)
    if emb.shape[0] != len(align_keys):
        raise ValueError(
            f"embeddings.npy row count ({emb.shape[0]}) must match alignment file ({len(align_keys)})"
        )

    id_to_row = {str(align_keys[i]): i for i in range(len(align_keys))}
    indices: list[int] = []
    missing: list[str] = []
    for cid in df[ID_COL].astype(str):
        idx = id_to_row.get(cid)
        if idx is None:
            missing.append(cid)
        else:
            indices.append(idx)
    if missing:
        raise KeyError(
            f"{len(missing)} comment_id(s) not found in embedding file(s). "
            f"Example: {missing[:3]}. Regenerate embeddings with bert.py on this CSV."
        )

    text_embeds = emb[indices].astype(np.float32, copy=False)
    demo = df[DEMO_COLS].to_numpy(dtype=np.float32)
    labels = df[LABEL_COL].to_numpy(dtype=np.float32)

    workers = df[WORKER_COL].astype(str).tolist()
    unique_workers = sorted(set(workers))
    worker_to_idx = {w: i for i, w in enumerate(unique_workers)}
    worker_idx = np.array([worker_to_idx[w] for w in workers], dtype=np.int64)
    comment_ids = df[ID_COL].astype(str).to_numpy()

    return text_embeds, demo, worker_idx, labels, worker_to_idx, comment_ids


def train_val_split(
    text: np.ndarray,
    demo: np.ndarray,
    worker: np.ndarray,
    labels: np.ndarray,
    comment_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    """Comment-level split: disjoint ``comment_id`` sets between train and val."""
    tr, va = comment_level_split_indices(comment_ids, labels, val_fraction, seed)

    def subset(idxs: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(text[idxs]),
            torch.from_numpy(demo[idxs]),
            torch.from_numpy(worker[idxs]),
            torch.from_numpy(labels[idxs]),
        )

    return subset(tr), subset(va)


def evaluate(
    model: DCN,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float, float | None]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        val_pbar = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs} [val]",
            leave=False,
            dynamic_ncols=True,
        )
        for text_emb, demo, worker, label in val_pbar:
            text_emb = text_emb.to(device)
            demo = demo.to(device)
            worker = worker.to(device)
            label = label.to(device)
            logits = model(text_emb, demo, worker)
            loss = F.binary_cross_entropy_with_logits(logits, label, reduction="sum")
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            total_loss += loss.item()
            total_correct += (preds == label).sum().item()
            total_count += label.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            avg_loss_so_far = total_loss / max(total_count, 1)
            acc_so_far = total_correct / max(total_count, 1)
            val_pbar.set_postfix(loss=f"{avg_loss_so_far:.4f}", acc=f"{acc_so_far:.4f}")

    avg_loss = total_loss / max(total_count, 1)
    acc = total_correct / max(total_count, 1)

    auc: float | None = None
    try:
        from sklearn.metrics import roc_auc_score

        y = np.concatenate(all_labels)
        p = np.concatenate(all_probs)
        if len(np.unique(y)) > 1:
            auc = float(roc_auc_score(y, p))
    except Exception:
        pass

    return avg_loss, acc, auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DCN on embeddings + demographics + worker")
    p.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMB)
    p.add_argument(
        "--embedding-alignment",
        "--comment-ids",
        type=Path,
        default=DEFAULT_EMB_ALIGNMENT,
        dest="embedding_alignment",
        help="NPY: one join key per embeddings row (e.g. comment_id). Not a model input.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-cross-layers", type=int, default=3)
    p.add_argument("--num-deep-layers", type=int, default=3)
    p.add_argument("--deep-layer-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--model",
        choices=["dcn", "dcnv2"],
        default="dcn",
        help="Model architecture to train.",
    )
    p.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/dcn_stigma.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    text_np, demo_np, worker_np, labels_np, worker_map, comment_ids_np = load_arrays(
        args.final_csv,
        args.embeddings,
        args.embedding_alignment,
    )
    num_workers_vocab = len(worker_map)

    train_ds, val_ds = train_val_split(
        text_np,
        demo_np,
        worker_np,
        labels_np,
        comment_ids_np,
        args.val_fraction,
        args.seed,
    )

    device = _pick_device()
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin,
    )

    model_cls = DCN if args.model == "dcn" else DCNv2
    model = model_cls(
        input_dim=INPUT_DIM,
        num_workers=num_workers_vocab,
        annotator_emb_dim=ANNOTATOR_EMB_DIM,
        text_dim=TEXT_EMB_DIM,
        demo_dim=DEMO_DIM,
        num_cross_layers=args.num_cross_layers,
        num_deep_layers=args.num_deep_layers,
        deep_layer_size=args.deep_layer_size,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs} [train]",
            leave=False,
            dynamic_ncols=True,
        )
        for text_emb, demo, worker, label in train_pbar:
            text_emb = text_emb.to(device)
            demo = demo.to(device)
            worker = worker.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(text_emb, demo, worker)
            loss = F.binary_cross_entropy_with_logits(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            running_loss += loss.item() * label.size(0)
            running_correct += (preds == label).sum().item()
            running_count += label.size(0)
            avg_loss_so_far = running_loss / max(running_count, 1)
            acc_so_far = running_correct / max(running_count, 1)
            train_pbar.set_postfix(loss=f"{avg_loss_so_far:.4f}", acc=f"{acc_so_far:.4f}")

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, device, epoch + 1, args.epochs)
        scheduler.step(val_loss)

        auc_str = f" AUC: {val_auc:.4f}" if val_auc is not None else ""
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}{auc_str}"
        )

        if val_loss < best_val:
            best_val = val_loss
            cfg = vars(args).copy()
            cfg["split_strategy"] = "comment_id"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "worker_to_idx": worker_map,
                    "config": cfg,
                },
                args.checkpoint,
            )
            print(f"  saved checkpoint to {args.checkpoint}")

    print("Training complete.")


if __name__ == "__main__":
    main()
