"""Comment-level split helpers for train/validation sets."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_cid_array(comment_ids: Sequence) -> np.ndarray:
    arr = np.asarray(comment_ids)
    if arr.dtype.kind in ("U", "S", "O"):
        return arr.astype(str, copy=False)
    return arr.astype(str)


def _per_comment_label(
    cids: np.ndarray,
    labels: np.ndarray,
    unique_cids: np.ndarray,
) -> np.ndarray:
    """Majority-vote label per comment, aligned to ``unique_cids`` order."""
    import pandas as pd

    s = pd.Series(np.asarray(labels, dtype=float))
    s.index = pd.Index(cids, name="cid")
    mean_per_cid = s.groupby(level=0).mean()
    vals = mean_per_cid.reindex(unique_cids).to_numpy()
    return (vals >= 0.5).astype(int)


def comment_level_split_indices(
    comment_ids: Sequence,
    labels: np.ndarray | None,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(train_row_idx, val_row_idx)`` with disjoint comment_id sets.

    Splits the set of unique comments (stratified by majority-vote per-comment
    label when both classes are present), then expands back to row indices into
    ``comment_ids``. The row-level class balance is approximate, but no comment
    appears in both splits.
    """
    import pandas as pd

    cids = _as_cid_array(comment_ids)
    n = len(cids)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    unique_cids = pd.Series(cids).drop_duplicates().to_numpy()

    strat = None
    if labels is not None:
        lbl = _per_comment_label(cids, np.asarray(labels), unique_cids)
        if len(np.unique(lbl)) > 1:
            strat = lbl

    idx_range = np.arange(len(unique_cids))
    try:
        from sklearn.model_selection import train_test_split

        tr_ix, va_ix = train_test_split(
            idx_range,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
            stratify=strat,
        )
    except ImportError:
        rng = np.random.RandomState(seed)
        perm = idx_range.copy()
        rng.shuffle(perm)
        n_val = max(1, int(len(unique_cids) * val_fraction))
        va_ix = perm[:n_val]
        tr_ix = perm[n_val:]

    val_cid_set = set(unique_cids[va_ix].tolist())
    val_mask = np.fromiter((c in val_cid_set for c in cids), dtype=bool, count=len(cids))
    val_idx = np.flatnonzero(val_mask).astype(np.int64)
    train_idx = np.flatnonzero(~val_mask).astype(np.int64)

    return train_idx, val_idx
