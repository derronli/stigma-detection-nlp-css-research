"""Run jury-based inference with pair-encoded BERTweet + DCN checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from bert_pair import (
    BATCH_SIZE as PAIR_BATCH_SIZE,
    PRIMARY_MAX_TOKENS_DEFAULT,
    _clean_text,
    _encode_pair_batch,
    _pick_device,
)
from model.dcn import DCN
from model.dcnv2 import DCNv2

PROJECT_ROOT = _SRC.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints_v2/dcnv2_multitext_pair.pt"

DEMO_DIM = 4

_ALL_JURY_TUPLES: list[tuple[int, int, int, int]] = [
    (a, b, c, d)
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
    for d in (0, 1)
]
_EXCLUDED_JURIES: set[tuple[int, int, int, int]] = {
    (1, 0, 1, 0),
    (1, 0, 1, 1),
}
JURY_TUPLES: list[tuple[int, int, int, int]] = [
    t for t in _ALL_JURY_TUPLES if t not in _EXCLUDED_JURIES
]


def _jury_label(tup: tuple[int, int, int, int]) -> str:
    return "".join(str(v) for v in tup)


def _load_checkpoint(
    path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, int], dict]:
    """Load a checkpoint saved by trainer_multitext.py."""
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)

    if not isinstance(blob, dict) or "model_state_dict" not in blob:
        raise ValueError(
            f"checkpoint at {path} missing 'model_state_dict'; "
            "was it saved by trainer_multitext.py?"
        )
    if "config" not in blob or "worker_to_idx" not in blob:
        raise ValueError(
            f"checkpoint at {path} missing 'config' or 'worker_to_idx'; "
            "regenerate with the current trainer_multitext.py."
        )

    cfg = blob["config"]
    worker_map = blob["worker_to_idx"]
    num_workers_vocab = len(worker_map)
    model_type = cfg.get("model", "dcnv2")
    text_emb_dim = int(cfg["text_emb_dim"])
    input_dim = int(cfg["input_dim"])
    annotator_emb_dim = input_dim - text_emb_dim - DEMO_DIM

    model_cls = DCN if model_type == "dcn" else DCNv2
    model = model_cls(
        input_dim=input_dim,
        num_workers=num_workers_vocab,
        annotator_emb_dim=annotator_emb_dim,
        text_dim=text_emb_dim,
        demo_dim=DEMO_DIM,
        num_cross_layers=int(cfg.get("num_cross_layers", 3)),
        num_deep_layers=int(cfg.get("num_deep_layers", 2)),
        deep_layer_size=int(cfg.get("deep_layer_size", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    missing, unexpected = model.load_state_dict(blob["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch — missing {missing}, unexpected {unexpected}"
        )
    model.eval()
    return model, worker_map, cfg


def _get_annotator_weight(model: torch.nn.Module) -> torch.Tensor:
    """Return the annotator / worker embedding weight matrix (depends on model variant)."""
    if hasattr(model, "annotator_emb"):
        return model.annotator_emb.weight
    if hasattr(model, "worker_emb"):
        return model.worker_emb.weight
    raise AttributeError("model has neither 'annotator_emb' nor 'worker_emb'")


def _model_cross_forward(
    model: torch.nn.Module,
    text_emb: torch.Tensor,
    demo: torch.Tensor,
    annotator_emb: torch.Tensor,
) -> torch.Tensor:
    """Replicate DCN / DCNv2 forward, but with a precomputed annotator embedding.

    This lets us marginalize the worker input (e.g. by passing the mean of the
    trained annotator embedding weights) without any model-code changes.
    """
    x = torch.cat([text_emb, demo, annotator_emb], dim=1)
    x0 = x
    x_cross = x
    for layer in model.cross_layers:
        x_cross = layer(x0, x_cross)
    x_deep = model.deep(x)
    x_cat = torch.cat([x_cross, x_deep], dim=1)
    head = getattr(model, "output", None) or getattr(model, "final_linear", None)
    if head is None:
        raise AttributeError("model has neither 'output' nor 'final_linear' head")
    logits = head(x_cat).squeeze(-1)
    return logits


def _encode_texts(
    comments: list[str],
    parents: list[str],
    tokenizer: AutoTokenizer,
    bertweet: AutoModel,
    device: torch.device,
    pooling: str,
    primary_max_tokens: int,
    batch_size: int,
) -> np.ndarray:
    hidden = bertweet.config.hidden_size
    out = np.zeros((len(comments), hidden), dtype=np.float32)
    for start in tqdm(
        range(0, len(comments), batch_size),
        desc=f"BERTweet pair [{pooling}]",
        dynamic_ncols=True,
    ):
        end = min(start + batch_size, len(comments))
        out[start:end] = _encode_pair_batch(
            comments[start:end],
            parents[start:end],
            tokenizer,
            bertweet,
            device,
            pooling,
            primary_max_tokens,
        )
    return out


def _score_one_jury(
    model: torch.nn.Module,
    text_emb_all: torch.Tensor,
    jury_demo: tuple[int, int, int, int],
    annotator_weight: torch.Tensor,
    worker_strategy: str,
    device: torch.device,
    scoring_batch_size: int,
) -> np.ndarray:
    """Return a ``[num_rows]`` numpy array of sigmoid probabilities for this jury."""
    n = text_emb_all.shape[0]
    demo_vec = torch.tensor(jury_demo, dtype=torch.float32, device=device)
    out = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        if worker_strategy == "mean_embedding":
            mean_annot = annotator_weight.mean(dim=0)  # [annot_dim]
            for start in range(0, n, scoring_batch_size):
                end = min(start + scoring_batch_size, n)
                bsz = end - start
                text_b = text_emb_all[start:end]
                demo_b = demo_vec.unsqueeze(0).expand(bsz, -1)
                annot_b = mean_annot.unsqueeze(0).expand(bsz, -1)
                logits = _model_cross_forward(model, text_b, demo_b, annot_b)
                out[start:end] = torch.sigmoid(logits).cpu().numpy()
        elif worker_strategy == "average_logits":
            # Average sigmoid probabilities across every known worker.
            num_workers = annotator_weight.shape[0]
            for start in range(0, n, scoring_batch_size):
                end = min(start + scoring_batch_size, n)
                bsz = end - start
                text_b = text_emb_all[start:end]
                demo_b = demo_vec.unsqueeze(0).expand(bsz, -1)
                prob_sum = torch.zeros(bsz, device=device)
                for w_idx in range(num_workers):
                    annot_b = annotator_weight[w_idx].unsqueeze(0).expand(bsz, -1)
                    logits = _model_cross_forward(model, text_b, demo_b, annot_b)
                    prob_sum += torch.sigmoid(logits)
                out[start:end] = (prob_sum / max(num_workers, 1)).cpu().numpy()
        else:
            raise ValueError(f"unknown worker_strategy: {worker_strategy}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Jury-majority inference for dcnv2_multitext_pair checkpoints.",
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument(
        "--comment-col",
        type=str,
        default="comment_text",
        help="Column name for the primary text (the comment being labeled).",
    )
    p.add_argument(
        "--parent-col",
        type=str,
        default="parent_text",
        help=(
            "Column name for the parent text (second segment in pair encoding). "
            "If the column is missing, it's treated as empty for every row."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Per-jury decision threshold applied to sigmoid probability.",
    )
    p.add_argument(
        "--worker-strategy",
        choices=["mean_embedding", "average_logits"],
        default="mean_embedding",
        help=(
            "How to marginalize the annotator / worker input. "
            "'mean_embedding' is fast (one forward per jury). "
            "'average_logits' averages over every trained worker (slower but "
            "more faithful to the training-time distribution)."
        ),
    )
    p.add_argument(
        "--tie-break",
        choices=["mean_score", "zero", "one"],
        default="mean_score",
        help=(
            "How to break a 7-7 tie across juries. 'mean_score' uses the "
            "average jury probability vs. --threshold (default). "
            "'zero' / 'one' are fixed tie-break labels."
        ),
    )
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument(
        "--primary-max-tokens",
        type=int,
        default=PRIMARY_MAX_TOKENS_DEFAULT,
        help="Must match what bert_pair.py used at training time.",
    )
    p.add_argument(
        "--encode-batch-size",
        type=int,
        default=PAIR_BATCH_SIZE,
        help="Batch size for BERTweet encoding.",
    )
    p.add_argument(
        "--score-batch-size",
        type=int,
        default=256,
        help="Batch size for DCN scoring (per jury).",
    )
    p.add_argument(
        "--include-per-jury",
        action="store_true",
        help="If set, append per-jury prediction columns (prob_<bits>, vote_<bits>).",
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
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError(f"--threshold must be in (0, 1), got {args.threshold}")

    device = _pick_device()
    print(f"Device: {device}")

    df = pd.read_csv(args.input_csv)
    if args.comment_col not in df.columns:
        raise ValueError(
            f"Input CSV is missing required column '{args.comment_col}'. "
            f"Available columns: {list(df.columns)}"
        )
    if args.parent_col not in df.columns:
        print(
            f"Note: input CSV has no '{args.parent_col}' column; "
            "encoding every row as a single segment."
        )
        df[args.parent_col] = ""

    comments_raw = df[args.comment_col].fillna("").astype(str).tolist()
    parents_raw = df[args.parent_col].fillna("").astype(str).tolist()
    comments = [_clean_text(t, args.replace_deleted_with_empty) for t in comments_raw]
    parents = [_clean_text(t, args.replace_deleted_with_empty) for t in parents_raw]

    model, worker_map, cfg = _load_checkpoint(args.checkpoint, device)
    annotator_weight = _get_annotator_weight(model).detach()
    expected_text_dim = int(cfg["text_emb_dim"])
    print(
        f"Loaded {cfg.get('model', 'dcnv2')} checkpoint: "
        f"text_dim={expected_text_dim}  workers={len(worker_map)}  "
        f"cross={cfg.get('num_cross_layers')} deep={cfg.get('num_deep_layers')}x{cfg.get('deep_layer_size')}"
    )

    # Encode texts once — reused across all juries.
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
    bertweet = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
    bertweet.eval()

    text_np = _encode_texts(
        comments,
        parents,
        tokenizer,
        bertweet,
        device,
        args.pooling,
        args.primary_max_tokens,
        args.encode_batch_size,
    )
    if text_np.shape[1] != expected_text_dim:
        raise ValueError(
            f"Encoded text dim ({text_np.shape[1]}) != checkpoint text_dim "
            f"({expected_text_dim}). Re-check --primary-max-tokens / --pooling "
            "or that this checkpoint was trained on bert_pair.py output."
        )
    text_tensor = torch.from_numpy(text_np).to(device)

    # Free BERTweet — not needed for the scoring phase.
    del bertweet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Score each jury. Collect per-jury probabilities then aggregate.
    n = len(df)
    num_juries = len(JURY_TUPLES)
    all_probs = np.zeros((num_juries, n), dtype=np.float32)

    jury_iter = tqdm(
        list(enumerate(JURY_TUPLES)),
        desc=f"Scoring juries ({args.worker_strategy})",
        dynamic_ncols=True,
    )
    for j_idx, jury in jury_iter:
        probs = _score_one_jury(
            model=model,
            text_emb_all=text_tensor,
            jury_demo=jury,
            annotator_weight=annotator_weight,
            worker_strategy=args.worker_strategy,
            device=device,
            scoring_batch_size=args.score_batch_size,
        )
        all_probs[j_idx] = probs

    # Per-jury binary votes at the chosen threshold.
    per_jury_votes = (all_probs >= args.threshold).astype(np.int32)  # [J, N]
    votes_positive = per_jury_votes.sum(axis=0)  # [N]
    mean_scores = all_probs.mean(axis=0)  # [N]

    # Majority label. Juries are even, so ties can occur at exactly J/2 positive.
    half = num_juries / 2.0
    final = np.zeros(n, dtype=np.int32)
    tie_mask = votes_positive.astype(float) == half
    final[votes_positive > half] = 1
    if args.tie_break == "mean_score":
        final[tie_mask] = (mean_scores[tie_mask] >= args.threshold).astype(np.int32)
    elif args.tie_break == "one":
        final[tie_mask] = 1
    # tie_break == 'zero' is the default (already zero-initialized).

    out_df = df.copy()
    out_df["mean_score"] = mean_scores
    out_df["votes_positive"] = votes_positive
    out_df["votes_total"] = num_juries
    out_df["final_label"] = final

    if args.include_per_jury:
        for j_idx, jury in enumerate(JURY_TUPLES):
            tag = _jury_label(jury)
            out_df[f"prob_{tag}"] = all_probs[j_idx]
            out_df[f"vote_{tag}"] = per_jury_votes[j_idx]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(
        f"\nWrote {n} rows to {args.output_csv}\n"
        f"  Juries voted      : {num_juries} "
        f"(excluded {sorted(_EXCLUDED_JURIES)})\n"
        f"  threshold         : {args.threshold}\n"
        f"  tie-break         : {args.tie_break}\n"
        f"  worker strategy   : {args.worker_strategy}\n"
        f"  final label == 1  : {int((final == 1).sum())} "
        f"({100.0 * (final == 1).mean():.1f}%)\n"
        f"  ties at {int(half)}-{int(half)}       : {int(tie_mask.sum())}"
    )


if __name__ == "__main__":
    main()
