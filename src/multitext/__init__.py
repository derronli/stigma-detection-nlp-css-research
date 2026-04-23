"""Utilities for multitext BERTweet embeddings + DCN training (additive pipeline)."""

from .text_fields import (
    ALLOWED_FIELDS,
    SELECTED_FIELDS,
    parse_fields_arg,
    total_text_dim,
)

__all__ = [
    "ALLOWED_FIELDS",
    "SELECTED_FIELDS",
    "parse_fields_arg",
    "total_text_dim",
]
