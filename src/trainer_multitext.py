"""Train DCN/DCNv2 on multitext embeddings and annotator metadata."""

from __future__ import annotations

import argparse
import json
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
DEFAULT_EMB = PROJECT_ROOT / "Data/embeddings_multitext_full/embeddings.npy"
DEFAULT_EMB_ALIGNMENT = PROJECT_ROOT / "Data/embeddings_multitext_full/comment_ids.npy"
DEFAULT_MANIFEST = PROJECT_ROOT / "Data/embeddings_multitext_full/manifest.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/dcnv2_multitext_full.pt"

DEMO_COLS = ["female", "knowsud", "black", "usesubstance"]
LABEL_COL = "stigma_value"
WORKER_COL = "worker_id"
ID_COL = "comment_id"

DEMO_DIM = len(DEMO_COLS)
ANNOTATOR_EMB_DIM = 8
BERTWEET_HIDDEN = 768


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_embedding_alignment_path(path: Path) -> Path | None:
    if path.exists():
        return path
    legacy = path.parent / "comment_ids.npy"
    if legacy.exists():
        return legacy
    return None


def resolve_included_indices_path(explicit: Path | None, embeddings_path: Path) -> Path | None:
    """Prefer explicit path; else embeddings_dir/included_indices.npy if present."""
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"--included-indices not found: {explicit}")
        return explicit
    auto = embeddings_path.parent / "included_indices.npy"
    if auto.exists():
        return auto
    return None


def load_arrays(
    final_csv: Path,
    embeddings_path: Path,
    embedding_alignment_path: Path,
    included_indices_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], int, np.ndarray]:
    """Return text_embeds [N,text_dim], demo, worker_idx, labels, worker_to_idx, text_dim, comment_ids.

    If ``included_indices.npy`` is provided or found next to ``embeddings_path``,
    ``df`` is ``final_csv.iloc[indices]`` so row counts match filtered embeddings.
    Otherwise row ``i`` matches ``final.csv`` row ``i`` (legacy full-table embeddings).
    """
    import pandas as pd

    df = pd.read_csv(final_csv)
    for c in [WORKER_COL, ID_COL, LABEL_COL, *DEMO_COLS]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {final_csv}")

    idx_path = resolve_included_indices_path(included_indices_path, embeddings_path)
    if idx_path is not None:
        indices = np.load(idx_path).astype(np.int64)
        if indices.ndim != 1:
            raise ValueError(f"included_indices must be 1-D, got shape {indices.shape}")
        if (indices < 0).any() or (indices >= len(df)).any():
            raise ValueError(
                f"included_indices out of range for final.csv (len={len(df)}): {idx_path}"
            )
        df = df.iloc[indices].reset_index(drop=True)

    emb = np.load(embeddings_path)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings array, got shape {emb.shape}")

    text_dim = emb.shape[1]
    if text_dim % BERTWEET_HIDDEN != 0:
        raise ValueError(
            f"embedding width {text_dim} is not a multiple of BERTweet hidden ({BERTWEET_HIDDEN})"
        )

    if emb.shape[0] != len(df):
        raise ValueError(
            f"Multitext embeddings rows ({emb.shape[0]}) must equal training rows ({len(df)}). "
            "Use the same final.csv as in bert_multitext; if embeddings used row filtering, "
            "ensure included_indices.npy is next to embeddings.npy (or pass --included-indices)."
        )

    align_file = resolve_embedding_alignment_path(embedding_alignment_path)
    if align_file is not None:
        align_keys = np.load(align_file, allow_pickle=True)
        if len(align_keys) == len(df):
            mism = np.sum(align_keys.astype(str) != df[ID_COL].astype(str).to_numpy())
            if mism:
                raise ValueError(
                    f"{mism} rows: comment_ids.npy does not match final.csv comment_id order. "
                    "Use embeddings generated for this final.csv."
                )

    text_embeds = emb.astype(np.float32, copy=False)
    demo = df[DEMO_COLS].to_numpy(dtype=np.float32)
    labels = df[LABEL_COL].to_numpy(dtype=np.float32)

    workers = df[WORKER_COL].astype(str).tolist()
    unique_workers = sorted(set(workers))
    worker_to_idx = {w: i for i, w in enumerate(unique_workers)}
    worker_idx = np.array([worker_to_idx[w] for w in workers], dtype=np.int64)
    comment_ids = df[ID_COL].astype(str).to_numpy()

    return text_embeds, demo, worker_idx, labels, worker_to_idx, text_dim, comment_ids


def maybe_warn_manifest(manifest_path: Path | None, text_dim: int) -> None:
    if manifest_path is None or not manifest_path.exists():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        m_dim = data.get("text_dim")
        if m_dim is not None and int(m_dim) != text_dim:
            print(
                f"Warning: manifest text_dim ({m_dim}) != embeddings width ({text_dim}). "
                "Wrong manifest or mismatched embedding file."
            )
    except Exception:
        pass


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
    model: torch.nn.Module,
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
    p = argparse.ArgumentParser(description="Train DCN on multitext BERTweet embeddings")
    p.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    p.add_argument("--embeddings", type=Path, default=DEFAULT_EMB)
    p.add_argument(
        "--embedding-alignment",
        "--comment-ids",
        type=Path,
        default=DEFAULT_EMB_ALIGNMENT,
        dest="embedding_alignment",
        help="Optional: comment_ids.npy for sanity check vs final.csv row order.",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Optional manifest.json from bert_multitext (warns if text_dim mismatches).",
    )
    p.add_argument(
        "--included-indices",
        type=Path,
        default=None,
        help=(
            "Optional NPY of integer row positions into final.csv (bert_multitext writes this). "
            "If omitted, uses embeddings_dir/included_indices.npy when present."
        ),
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
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
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    text_np, demo_np, worker_np, labels_np, worker_map, text_emb_dim, comment_ids_np = load_arrays(
        args.final_csv,
        args.embeddings,
        args.embedding_alignment,
        args.included_indices,
    )
    maybe_warn_manifest(args.manifest, text_emb_dim)

    input_dim = text_emb_dim + DEMO_DIM + ANNOTATOR_EMB_DIM
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
        input_dim=input_dim,
        num_workers=num_workers_vocab,
        annotator_emb_dim=ANNOTATOR_EMB_DIM,
        text_dim=text_emb_dim,
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
            cfg["text_emb_dim"] = text_emb_dim
            cfg["input_dim"] = input_dim
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
