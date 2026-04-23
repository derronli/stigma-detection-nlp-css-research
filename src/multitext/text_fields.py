"""Field configuration for multitext embedding generation."""

from __future__ import annotations

BERTWEET_HIDDEN_SIZE = 768

# Columns expected after merging final.csv with fetched_context_data.csv on comment_id.
ALLOWED_FIELDS = frozenset({"comment_text", "parent_text", "parent_post_text"})

# Default: comment + Reddit parent reply + submission text.
SELECTED_FIELDS: tuple[str, ...] = (
    "comment_text",
)


def total_text_dim(num_fields: int, hidden_size: int = BERTWEET_HIDDEN_SIZE) -> int:
    return hidden_size * num_fields


def parse_fields_arg(fields_csv: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse CLI `--fields comment_text,parent_text`; None or empty uses default."""
    if not fields_csv or not str(fields_csv).strip():
        return default
    parts = tuple(s.strip() for s in fields_csv.split(",") if s.strip())
    unknown = set(parts) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unknown field(s): {sorted(unknown)}. Allowed: {sorted(ALLOWED_FIELDS)}")
    if not parts:
        return default
    return parts
