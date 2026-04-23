"""Generate multitext BERTweet embeddings by field-wise concatenation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from multitext.merge_final_context import merge_final_with_context
from multitext.text_fields import (
    SELECTED_FIELDS,
    parse_fields_arg,
    total_text_dim,
)

INCLUDED_INDICES_FILENAME = "included_indices.npy"

PROJECT_ROOT = _SRC.parent
DEFAULT_FINAL_CSV = PROJECT_ROOT / "Data/Dan annotations5k 04202022/final.csv"
DEFAULT_CONTEXT_CSV = PROJECT_ROOT / "Data/Dan annotations5k 04202022/fetched_context_data.csv"
OUTPUT_ROOT = PROJECT_ROOT / "Data/embeddings_multitext_base_unfiltered"

MAX_LENGTH = 128
BATCH_SIZE = 32


def mask_complete_text_rows(merged, fields: tuple[str, ...]) -> np.ndarray:
    """Boolean array: True where every selected column has non-empty text after strip."""
    mask = np.ones(len(merged), dtype=bool)
    for f in fields:
        col = merged[f].fillna("").astype(str).str.strip()
        mask &= col.ne("").to_numpy()
    return mask


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_berttweet_embeddings(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
) -> np.ndarray:
    """CLS embeddings [batch, hidden]."""
    with torch.no_grad():
        encoded = tokenizer(
            [str(t) for t in texts],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = model(**encoded)
        return output.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BERTweet multitext embeddings for DCN")
    p.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    p.add_argument("--context-csv", type=Path, default=DEFAULT_CONTEXT_CSV)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Writes embeddings.npy, comment_ids.npy, included_indices.npy, manifest.json here.",
    )
    p.add_argument(
        "--fields",
        type=str,
        default="",
        help=f"Comma-separated subset of comment_text,parent_text,parent_post_text. "
        f"Default: from multitext/text_fields.py SELECTED_FIELDS.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="After filtering, encode only the first N rows (smoke tests).",
    )
    p.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Drop rows where any selected field is empty/missing.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fields = parse_fields_arg(args.fields, SELECTED_FIELDS)

    merged = merge_final_with_context(args.final_csv, args.context_csv)

    for f in fields:
        if f not in merged.columns:
            raise ValueError(f"Column '{f}' missing after merge. Allowed context columns come from context CSV.")

    skip_incomplete = args.skip_incomplete
    if skip_incomplete:
        complete = mask_complete_text_rows(merged, fields)
        n_drop = int((~complete).sum())
        merged = merged.loc[complete].reset_index(drop=True)
        included_indices = np.flatnonzero(complete).astype(np.int64)
        if len(merged) == 0:
            raise ValueError(
                "No rows left after skipping incomplete rows (empty/missing values in selected fields). "
                "Relax fields or use --no-skip-incomplete."
            )
        print(f"Skipping {n_drop} incomplete row(s); keeping {len(merged)} row(s).")
    else:
        included_indices = np.arange(len(merged), dtype=np.int64)

    if args.max_rows is not None:
        merged = merged.iloc[: args.max_rows].copy()
        included_indices = included_indices[: args.max_rows]

    total = len(merged)
    if total == 0:
        raise ValueError("No rows after merge / filter.")

    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
    model = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
    model.eval()
    hidden = model.config.hidden_size
    text_dim = total_text_dim(len(fields), hidden)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_out = args.output_dir / "embeddings.npy"
    align_out = args.output_dir / "comment_ids.npy"
    indices_out = args.output_dir / INCLUDED_INDICES_FILENAME
    manifest_out = args.output_dir / "manifest.json"

    all_embeddings = np.zeros((total, text_dim), dtype=np.float32)
    comment_ids = merged["comment_id"].astype(object).to_numpy()

    offset = 0
    for fi, field in enumerate(fields):
        col_emb = np.zeros((total, hidden), dtype=np.float32)
        texts = merged[field].fillna("").astype(str).tolist()
        for start in tqdm(range(0, total, BATCH_SIZE), desc=f"BERTweet [{field}]"):
            end = min(start + BATCH_SIZE, total)
            batch_emb = get_berttweet_embeddings(texts[start:end], tokenizer, model, device)
            col_emb[start:end] = batch_emb
        all_embeddings[:, offset : offset + hidden] = col_emb
        offset += hidden

    np.save(emb_out, all_embeddings)
    np.save(align_out, comment_ids)
    np.save(indices_out, included_indices)

    manifest = {
        "fields": list(fields),
        "hidden_size": hidden,
        "text_dim": text_dim,
        "final_csv": str(args.final_csv.resolve()),
        "context_csv": str(args.context_csv.resolve()),
        "num_rows": total,
        "embedding_shape": list(all_embeddings.shape),
        "skip_incomplete_rows": skip_incomplete,
        "included_indices_file": INCLUDED_INDICES_FILENAME,
        "included_indices_are_final_csv_positions": True,
        "note": (
            "Row j of embeddings.npy aligns with final.csv row included_indices[j]. "
            "trainer_multitext subsets final.csv using included_indices.npy when present."
        ),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Done. Saved multitext embeddings:\n"
        f"  {emb_out}  shape {all_embeddings.shape}\n"
        f"  {align_out}\n"
        f"  {indices_out}\n"
        f"  {manifest_out}\n"
        f"Fields (order): {fields}"
    )


if __name__ == "__main__":
    main()
