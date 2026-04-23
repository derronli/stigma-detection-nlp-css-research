"""Stream-filter Reddit comment dumps to the fields used by this project."""

import argparse
import csv
import json
import logging.handlers
import os
import re
import sys
import traceback
from datetime import datetime

import zstandard

# ---------------------------------------------------------------------------
# User configuration. Edit these.
# ---------------------------------------------------------------------------

# Input zst file *or* directory containing multiple zst files. If a directory
# is given, each ``.zst`` inside is processed and the output is a directory.
INPUT_FILE = "/Users/derronli/Downloads/reddit/subreddits25/REDDITORSINRECOVERY_comments.zst"  # e.g. r"/path/to/reddit_comments_2015_onwards"
# Output path (no extension; the script appends the right extension).
OUTPUT_FILE = "/Users/derronli/Downloads/reddit/processed/redditorsinrecovery_comments"  # e.g. r"/path/to/filtered_drug_comments"

# ndjson  : newline-delimited JSON. RECOMMENDED for this pipeline.
# zst     : zstandard-compressed ndjson (chainable with other pushshift
#           scripts; about 10-15x smaller on disk).
# csv     : CSV (lossy for comment bodies with newlines / commas; use only
#           for quick eyeballing).
OUTPUT_FORMAT = "zst"

# Only keep comments with created_utc >= FROM_DATE.
FROM_DATE = datetime(2015, 1, 1)
# Upper bound (sanity guard; ``datetime(2030, 12, 31)`` effectively disables it).
TO_DATE = datetime(2030, 12, 31)

# Substance / drug keywords. Matched case-insensitively against ``body``.
# "opiod" is kept alongside "opioid" because the most common misspelling is
# cheap to catch. Remove whichever you don't want.
KEYWORDS: list[str] = [
    "drug",
    "cocaine",
    "heroin",
    "lsd",
    "marijuana",
    "meth",
    "opiate",
    "opioid",
    "opiod",
    "shrooms",
    "weed",
]

# If True, match keywords as whole words (\bword\b), not substrings.
# Higher precision, slightly lower recall (e.g. ``drugs`` still matches due
# to the trailing boundary; ``drugstore`` no longer matches).
USE_WORD_BOUNDARY = True

# If True, drop comments whose body is literally "[removed]" or "[deleted]"
# (case-insensitive, stripped). Strongly recommended; these rows have no
# text signal for the model.
DROP_DELETED_REMOVED = True

# Projected output fields and how they map from a Pushshift comment record.
# Kept intentionally small but includes cheap context fields you're likely
# to want during downstream analysis.
#
# Mapping: output key -> pushshift field name. ``None`` means synthesized
# (see ``_project_comment`` below).
FIELD_MAP: dict[str, str | None] = {
    "comment_id":   "id",           # bare id (no "t1_" prefix)
    "comment_text": "body",
    "parent_id":    "parent_id",    # keeps the "t1_"/"t3_" prefix
    "link_id":      "link_id",      # "t3_<submission_id>"; cheap to keep
    "subreddit":    "subreddit",
    "author":       "author",
    "score":        "score",
    "created_utc":  "created_utc",  # epoch seconds; int, easy to filter
    "created_date": None,           # derived ISO date string for convenience
}

# Log every line that fails to parse. Turn off for expected-dirty dumps.
WRITE_BAD_LINES = True

# ---------------------------------------------------------------------------
# Internals. You probably don't need to edit below this line.
# ---------------------------------------------------------------------------

log = logging.getLogger("bot")
log.setLevel(logging.INFO)
_log_formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
_log_str_handler = logging.StreamHandler()
_log_str_handler.setFormatter(_log_formatter)
log.addHandler(_log_str_handler)
if not os.path.exists("logs"):
    os.makedirs("logs")
_log_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join("logs", "filter_reddit_drug_comments.log"),
    maxBytes=1024 * 1024 * 16,
    backupCount=5,
)
_log_file_handler.setFormatter(_log_formatter)
log.addHandler(_log_file_handler)


def _build_matcher(keywords: list[str], word_boundary: bool):
    """Return a ``body -> bool`` matcher.

    Compiles once up front; hot-loop cost is one regex search or a handful
    of ``in`` checks per comment.
    """
    kws = [k.lower() for k in keywords if k.strip()]
    if not kws:
        raise ValueError("KEYWORDS is empty")

    if word_boundary:
        pattern = r"\b(?:" + "|".join(re.escape(k) for k in kws) + r")\b"
        rx = re.compile(pattern, flags=re.IGNORECASE)

        def match(body: str) -> bool:
            return rx.search(body) is not None

        return match

    # Substring matcher — lowercasing once per body is faster than repeated
    # case-insensitive ``.find`` calls for ~10 keywords.
    def match(body: str) -> bool:
        low = body.lower()
        for k in kws:
            if k in low:
                return True
        return False

    return match


def _noop_keyword_match(_body: str) -> bool:
    """Used when ``--no-keyword-filter`` is set."""
    return True


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream-filter Reddit comment zst dumps (see module docstring).",
    )
    p.add_argument(
        "--no-keyword-filter",
        action="store_true",
        help=(
            "Do not require drug/substance keywords in body; keep every comment "
            "that passes the date range and optional [deleted]/[removed] drop."
        ),
    )
    return p.parse_args()


def _project_comment(obj: dict) -> dict:
    """Build the small output dict for one comment."""
    out: dict = {}
    for out_key, src in FIELD_MAP.items():
        if src is None:
            continue
        out[out_key] = obj.get(src)

    # Synthesize derived fields.
    if "created_date" in FIELD_MAP:
        ts = obj.get("created_utc")
        if ts is not None:
            try:
                out["created_date"] = datetime.utcfromtimestamp(int(ts)).strftime(
                    "%Y-%m-%d"
                )
            except (TypeError, ValueError):
                out["created_date"] = None
        else:
            out["created_date"] = None
    return out


_DELETED_PLACEHOLDERS = {"[deleted]", "[removed]"}


def _is_deleted_placeholder(body: str) -> bool:
    if body is None:
        return True
    return body.strip().lower() in _DELETED_PLACEHOLDERS


def _write_line_zst(handle, line: str) -> None:
    handle.write(line.encode("utf-8"))
    handle.write(b"\n")


def _write_line_json(handle, obj: dict) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False))
    handle.write("\n")


def _write_line_csv(writer, obj: dict, columns: list[str]) -> None:
    writer.writerow([obj.get(c, "") for c in columns])


def _read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(
                f"Unable to decode frame after reading {bytes_read:,} bytes"
            )
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return _read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def _read_lines_zst(file_name: str):
    with open(file_name, "rb") as file_handle:
        buffer = ""
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = _read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip(), file_handle.tell()
            buffer = lines[-1]
        reader.close()


def _open_output(output_path: str, output_format: str):
    """Return (handle, csv_writer_or_None)."""
    if output_format == "zst":
        handle = zstandard.ZstdCompressor().stream_writer(open(output_path, "wb"))
        return handle, None
    if output_format == "ndjson":
        handle = open(output_path, "w", encoding="utf-8")
        return handle, None
    if output_format == "csv":
        handle = open(output_path, "w", encoding="utf-8", newline="")
        writer = csv.writer(handle)
        writer.writerow(list(FIELD_MAP.keys()))
        return handle, writer
    raise ValueError(f"Unsupported output format {output_format}")


def process_file(
    input_file: str,
    output_file: str,
    output_format: str,
    matcher,
) -> None:
    ext = {"zst": "zst", "ndjson": "ndjson", "csv": "csv"}[output_format]
    output_path = f"{output_file}.{ext}"
    log.info(f"Input: {input_file} : Output: {output_path}")

    handle, writer = _open_output(output_path, output_format)
    columns = list(FIELD_MAP.keys())

    file_size = os.stat(input_file).st_size
    matched_lines = 0
    total_lines = 0
    bad_lines = 0
    skipped_date = 0
    skipped_nomatch = 0
    skipped_deleted = 0
    created = None

    try:
        for line, file_bytes_processed in _read_lines_zst(input_file):
            total_lines += 1
            if total_lines % 100_000 == 0:
                pct = (file_bytes_processed / file_size) * 100 if file_size else 0.0
                log.info(
                    f"{created.strftime('%Y-%m-%d') if created else '?'} : "
                    f"total={total_lines:,} kept={matched_lines:,} "
                    f"bad={bad_lines:,} "
                    f"{file_bytes_processed:,}B ({pct:.1f}%)"
                )

            try:
                obj = json.loads(line)
                # Date filter first — cheapest, most selective over time.
                ts = obj.get("created_utc")
                if ts is None:
                    skipped_date += 1
                    continue
                created = datetime.utcfromtimestamp(int(ts))
                if created < FROM_DATE or created > TO_DATE:
                    skipped_date += 1
                    continue

                body = obj.get("body")
                if body is None:
                    skipped_deleted += 1
                    continue
                if DROP_DELETED_REMOVED and _is_deleted_placeholder(body):
                    skipped_deleted += 1
                    continue

                if not matcher(body):
                    skipped_nomatch += 1
                    continue

                out_obj = _project_comment(obj)
                matched_lines += 1

                if output_format == "zst":
                    # For zst we recompress the *projected* JSON, not the
                    # original line — that's the whole point of filtering
                    # down to a small field set.
                    _write_line_zst(handle, json.dumps(out_obj, ensure_ascii=False))
                elif output_format == "ndjson":
                    _write_line_json(handle, out_obj)
                elif output_format == "csv":
                    _write_line_csv(writer, out_obj, columns)

            except (KeyError, json.JSONDecodeError, ValueError) as err:
                bad_lines += 1
                if WRITE_BAD_LINES:
                    log.warning(f"Skipping bad line ({err.__class__.__name__}): {err}")
    finally:
        handle.close()

    log.info(
        f"Done {input_file}: total={total_lines:,} kept={matched_lines:,} "
        f"bad={bad_lines:,}  skipped[date={skipped_date:,} nomatch={skipped_nomatch:,} "
        f"deleted={skipped_deleted:,}]"
    )


def _collect_input_files(input_file: str, output_file: str) -> list[tuple[str, str]]:
    """Return a list of ``(input_path, output_path_stem)`` pairs."""
    if os.path.isdir(input_file):
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        pairs = []
        for fname in os.listdir(input_file):
            full = os.path.join(input_file, fname)
            if os.path.isfile(full) and fname.endswith(".zst"):
                stem = os.path.splitext(os.path.splitext(os.path.basename(fname))[0])[0]
                pairs.append((full, os.path.join(output_file, stem)))
        return pairs
    return [(input_file, output_file)]


def _main() -> None:
    cli = _parse_cli()

    if not INPUT_FILE or not OUTPUT_FILE:
        log.error(
            "Set INPUT_FILE and OUTPUT_FILE at the top of this script before running."
        )
        sys.exit(2)

    if OUTPUT_FORMAT not in {"zst", "ndjson", "csv"}:
        log.error(f"Unsupported OUTPUT_FORMAT: {OUTPUT_FORMAT}")
        sys.exit(2)

    if cli.no_keyword_filter:
        matcher = _noop_keyword_match
        log.info("Keyword filtering: OFF (--no-keyword-filter)")
    else:
        matcher = _build_matcher(KEYWORDS, USE_WORD_BOUNDARY)
        log.info(
            f"Keyword filtering: ON — mode "
            f"{'word-boundary regex' if USE_WORD_BOUNDARY else 'substring'}"
        )
        log.info(f"Keywords ({len(KEYWORDS)}): {','.join(KEYWORDS)}")
    log.info(
        f"Date range: {FROM_DATE.strftime('%Y-%m-%d')} -> {TO_DATE.strftime('%Y-%m-%d')}"
    )
    log.info(f"Drop [deleted]/[removed] bodies: {DROP_DELETED_REMOVED}")
    log.info(f"Output format: {OUTPUT_FORMAT}")
    log.info(f"Output fields: {', '.join(FIELD_MAP.keys())}")

    input_files = _collect_input_files(INPUT_FILE, OUTPUT_FILE)
    log.info(f"Processing {len(input_files)} file(s)")

    for file_in, file_out in input_files:
        try:
            process_file(file_in, file_out, OUTPUT_FORMAT, matcher)
        except Exception as err:
            log.warning(f"Error processing {file_in}: {err}")
            log.warning(traceback.format_exc())


if __name__ == "__main__":
    _main()
