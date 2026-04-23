"""Generate pair-encoded BERTweet embeddings for comment and parent text."""

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

INCLUDED_INDICES_FILENAME = "included_indices.npy"

PROJECT_ROOT = _SRC.parent
DEFAULT_FINAL_CSV = PROJECT_ROOT / "Data/Dan annotations5k 04202022/final.csv"
DEFAULT_CONTEXT_CSV = PROJECT_ROOT / "Data/Dan annotations5k 04202022/fetched_context_data.csv"
OUTPUT_ROOT = PROJECT_ROOT / "Data/embeddings_multitext_pair"

MAX_LENGTH = 128
BATCH_SIZE = 32
DELETED_PLACEHOLDERS = {"[deleted]", "[removed]"}

PRIMARY_MAX_TOKENS_DEFAULT = 96


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clean_text(raw: str, replace_deleted_with_empty: bool) -> str:
    """Normalize Reddit placeholder tokens to empty strings when requested."""
    s = str(raw) if raw is not None else ""
    if replace_deleted_with_empty:
        stripped = s.strip()
        if stripped in DELETED_PLACEHOLDERS:
            return ""
    return s


def _build_pair_inputs(
    primary_texts: list[str],
    secondary_texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    primary_max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manually build pair-encoded (input_ids, attention_mask) for BERTweet.

    BERTweet ships a slow ``BertweetTokenizer`` which silently ignores
    ``truncation="only_second"``; pair inputs then overflow the 130-position
    table. We side-step this by tokenizing each side separately, budgeting a
    generous slice of ``max_length`` for the primary field, and letting the
    secondary field fill whatever is left, then assembling special tokens via
    ``build_inputs_with_special_tokens``. Rows with empty secondary degrade
    cleanly to single-sequence encoding (``[CLS] primary [SEP]``).
    """
    num_specials_pair = tokenizer.num_special_tokens_to_add(pair=True)
    num_specials_single = tokenizer.num_special_tokens_to_add(pair=False)
    id_lists: list[list[int]] = []
    for p_text, s_text in zip(primary_texts, secondary_texts):
        p_ids = tokenizer.encode(p_text, add_special_tokens=False)
        s_text_stripped = s_text.strip() if s_text else ""
        if not s_text_stripped:
            budget = max_length - num_specials_single
            p_ids = p_ids[:budget]
            combined = tokenizer.build_inputs_with_special_tokens(p_ids)
        else:
            s_ids = tokenizer.encode(s_text_stripped, add_special_tokens=False)
            budget = max_length - num_specials_pair
            p_cap = min(len(p_ids), primary_max_tokens, budget)
            p_ids = p_ids[:p_cap]
            s_budget = max(budget - len(p_ids), 0)
            s_ids = s_ids[:s_budget]
            combined = tokenizer.build_inputs_with_special_tokens(p_ids, s_ids)
        if len(combined) > max_length:
            combined = combined[:max_length]
        id_lists.append(combined)

    batch_len = max((len(x) for x in id_lists), default=0)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full((len(id_lists), batch_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(id_lists), batch_len), dtype=torch.long)
    for i, ids in enumerate(id_lists):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, : len(ids)] = 1
    return input_ids, attention


def _encode_pair_batch(
    primary: list[str],
    secondary: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    pooling: str,
    primary_max_tokens: int,
) -> np.ndarray:
    """Return an array ``[len(primary), hidden]`` of pooled embeddings."""
    input_ids, attention = _build_pair_inputs(
        primary, secondary, tokenizer, MAX_LENGTH, primary_max_tokens
    )
    input_ids = input_ids.to(device)
    attention = attention.to(device)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention)
        hidden = output.last_hidden_state

        if pooling == "cls":
            pooled = hidden[:, 0, :]
        elif pooling == "mean":
            mask = attention.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = summed / denom
        else:
            raise ValueError(f"unknown pooling: {pooling}")

        return pooled.cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pair-encoded BERTweet embeddings (comment+parent)")
    p.add_argument("--final-csv", type=Path, default=DEFAULT_FINAL_CSV)
    p.add_argument("--context-csv", type=Path, default=DEFAULT_CONTEXT_CSV)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help="Writes embeddings.npy, comment_ids.npy, included_indices.npy, manifest.json.",
    )
    p.add_argument(
        "--primary-field",
        type=str,
        default="comment_text",
        help="Field tokenized as the first segment (kept when truncation is needed).",
    )
    p.add_argument(
        "--secondary-field",
        type=str,
        default="parent_text",
        help="Field tokenized as the second segment (truncated first when over MAX_LENGTH).",
    )
    p.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
        help="Masked-mean pooling (recommended) or CLS pooling.",
    )
    p.add_argument(
        "--replace-deleted-with-empty",
        action="store_true",
        default=True,
        help="Treat '[deleted]' / '[removed]' placeholders as empty strings (default: True).",
    )
    p.add_argument(
        "--no-replace-deleted-with-empty",
        dest="replace_deleted_with_empty",
        action="store_false",
        help="Encode '[deleted]' / '[removed]' verbatim instead of blanking them.",
    )
    p.add_argument(
        "--skip-incomplete",
        action="store_true",
        help=(
            "Drop rows where primary OR secondary is empty/whitespace "
            "(after placeholder normalization). Default: keep all rows; "
            "empty sides are passed through as empty strings."
        ),
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="After filtering, encode only the first N rows (smoke tests).",
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument(
        "--primary-max-tokens",
        type=int,
        default=PRIMARY_MAX_TOKENS_DEFAULT,
        help=(
            "Upper bound (in BPE tokens) reserved for the primary field in the pair "
            "encoding. BERTweet's slow tokenizer does not honor truncation='only_second', "
            "so we pre-truncate the secondary side manually. With MAX_LENGTH=128 and 4 "
            "special tokens, content budget is 124; default 96 keeps most comments intact "
            "and leaves ~28 tokens of parent context."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    merged = merge_final_with_context(args.final_csv, args.context_csv)
    for f in (args.primary_field, args.secondary_field):
        if f not in merged.columns:
            raise ValueError(
                f"Column '{f}' missing after merge. Check --primary-field / --secondary-field."
            )

    primary_raw = merged[args.primary_field].fillna("").astype(str).tolist()
    secondary_raw = merged[args.secondary_field].fillna("").astype(str).tolist()
    primary = [_clean_text(t, args.replace_deleted_with_empty) for t in primary_raw]
    secondary = [_clean_text(t, args.replace_deleted_with_empty) for t in secondary_raw]

    n_rows_in = len(merged)
    n_primary_empty = sum(1 for t in primary if not t.strip())
    n_secondary_empty = sum(1 for t in secondary if not t.strip())

    if args.skip_incomplete:
        mask = np.array(
            [bool(p.strip()) and bool(s.strip()) for p, s in zip(primary, secondary)],
            dtype=bool,
        )
        n_drop = int((~mask).sum())
        included_indices = np.flatnonzero(mask).astype(np.int64)
        merged = merged.loc[mask].reset_index(drop=True)
        primary = [primary[i] for i in included_indices.tolist()]
        secondary = [secondary[i] for i in included_indices.tolist()]
        print(f"Skipping {n_drop} incomplete row(s); keeping {len(merged)} row(s).")
    else:
        included_indices = np.arange(len(merged), dtype=np.int64)

    if args.max_rows is not None:
        merged = merged.iloc[: args.max_rows].copy()
        included_indices = included_indices[: args.max_rows]
        primary = primary[: args.max_rows]
        secondary = secondary[: args.max_rows]

    total = len(merged)
    if total == 0:
        raise ValueError("No rows after merge / filter.")

    device = _pick_device()
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    model = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
    model.eval()
    hidden = model.config.hidden_size

    args.output_dir.mkdir(parents=True, exist_ok=True)
    emb_out = args.output_dir / "embeddings.npy"
    align_out = args.output_dir / "comment_ids.npy"
    indices_out = args.output_dir / INCLUDED_INDICES_FILENAME
    manifest_out = args.output_dir / "manifest.json"

    all_embeddings = np.zeros((total, hidden), dtype=np.float32)
    comment_ids = merged["comment_id"].astype(object).to_numpy()

    for start in tqdm(range(0, total, args.batch_size), desc=f"BERTweet pair [{args.pooling}]"):
        end = min(start + args.batch_size, total)
        all_embeddings[start:end] = _encode_pair_batch(
            primary[start:end],
            secondary[start:end],
            tokenizer,
            model,
            device,
            args.pooling,
            args.primary_max_tokens,
        )

    np.save(emb_out, all_embeddings)
    np.save(align_out, comment_ids)
    np.save(indices_out, included_indices)

    manifest = {
        "encoder": "vinai/bertweet-base",
        "encoding": "pair",
        "primary_field": args.primary_field,
        "secondary_field": args.secondary_field,
        "truncation": "manual_secondary_preferential",
        "pooling": args.pooling,
        "max_length": MAX_LENGTH,
        "primary_max_tokens": int(args.primary_max_tokens),
        "replace_deleted_with_empty": bool(args.replace_deleted_with_empty),
        "fields": [f"{args.primary_field}+{args.secondary_field}"],
        "hidden_size": hidden,
        "text_dim": hidden,
        "final_csv": str(args.final_csv.resolve()),
        "context_csv": str(args.context_csv.resolve()),
        "num_rows_input": int(n_rows_in),
        "num_rows": int(total),
        "embedding_shape": list(all_embeddings.shape),
        "skip_incomplete_rows": bool(args.skip_incomplete),
        "empty_primary_rows": int(n_primary_empty),
        "empty_secondary_rows": int(n_secondary_empty),
        "included_indices_file": INCLUDED_INDICES_FILENAME,
        "included_indices_are_final_csv_positions": True,
        "note": (
            "Row j of embeddings.npy aligns with final.csv row included_indices[j]. "
            "trainer_multitext subsets final.csv using included_indices.npy when present. "
            "Pair encoding allows cross-attention between primary and secondary fields; "
            "text_dim equals hidden_size (single pooled vector per row)."
        ),
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Done. Saved pair-encoded embeddings:\n"
        f"  {emb_out}  shape {all_embeddings.shape}\n"
        f"  {align_out}\n"
        f"  {indices_out}\n"
        f"  {manifest_out}\n"
        f"Primary   : {args.primary_field}  (kept on truncation)\n"
        f"Secondary : {args.secondary_field}  (truncated first)\n"
        f"Pooling   : {args.pooling}\n"
        f"Empty primary rows   : {n_primary_empty} / {n_rows_in}\n"
        f"Empty secondary rows : {n_secondary_empty} / {n_rows_in}"
    )


if __name__ == "__main__":
    main()
