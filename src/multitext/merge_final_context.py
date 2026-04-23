"""Merge annotation CSV with fetched context CSV (left join on comment_id)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CONTEXT_TEXT_COLUMNS = ["comment_text", "parent_text", "parent_post_text"]


def merge_final_with_context(final_csv: Path, context_csv: Path) -> pd.DataFrame:
    """
    Left join final.csv rows with context rows on comment_id.

    training rows follow final.csv row order one-to-one (authoritative length).
    Context duplicates are collapsed with keep=\"first\".
    Missing context cells are filled with empty strings for text columns.
    """
    final = pd.read_csv(final_csv)
    ctx = pd.read_csv(context_csv)
    if "comment_id" not in final.columns:
        raise ValueError(f"{final_csv} must contain comment_id")
    if "comment_id" not in ctx.columns:
        raise ValueError(f"{context_csv} must contain comment_id")

    ctx = ctx.drop_duplicates(subset=["comment_id"], keep="first")
    merged = final.merge(ctx, on="comment_id", how="left", suffixes=("", "_ctx_dup"))

    for col in CONTEXT_TEXT_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
        else:
            merged[col] = merged[col].fillna("").astype(str)

    return merged
