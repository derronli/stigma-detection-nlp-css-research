"""Build a CSV with comment text plus fetched parent text."""

import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

# === CONFIGURE HERE ===
# Any mix of individual ``.ndjson`` files and/or directories (every
# ``.ndjson`` directly inside a listed directory is included).
INPUT_PATHS: List[str] = [
    "./reddit/drugs_comments.ndjson",
    "./reddit/addiction_comments.ndjson",
    "./reddit/redditorsinrecovery_comments.ndjson",
]
OUTPUT_CSV = "./reddit/drug_subreddits.csv"

USER_AGENT = "script:reddit_comment_fetcher by u/OkEntry1846"
INFO_URL = "https://www.reddit.com/api/info.json"
MAX_IDS_PER_REQUEST = 100

MAX_RETRIES = 5
INITIAL_DELAY = 5
BACKOFF_FACTOR = 2

BATCH_DELAY = 2
LONG_DELAY = 30
LONG_DELAY_EVERY = 100


def submission_text(data: Dict[str, Any]) -> str:
    title = (data.get("title") or "").strip()
    selftext = (data.get("selftext") or "").strip()
    if selftext:
        return f"{title}\n\n{selftext}".strip() if title else selftext
    return title


def thing_text(kind: str, data: Dict[str, Any]) -> str:
    if kind == "t1":
        return (data.get("body") or "").strip()
    if kind == "t3":
        return submission_text(data)
    return ""


def normalize_fullname(s: Optional[Any]) -> Optional[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).strip()
    return t if t else None


def as_clean_str(v: Any) -> str:
    """Return a normalized string for CSV output.

    Accepts strings, numbers, booleans, and other JSON-native values.
    ``None`` / NaN-like values become ``""``.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def chunks(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def fetch_info_batch(ids: List[str]) -> Dict[str, str]:
    """Map fullname (t1_/t3_) -> body/title+selftext. Missing ids omitted."""
    if not ids:
        return {}
    headers = {"User-Agent": USER_AGENT}
    delay = INITIAL_DELAY
    params = {"id": ",".join(ids)}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(INFO_URL, headers=headers, params=params, timeout=60)
            if resp.status_code == 200:
                payload = resp.json()
                out: Dict[str, str] = {}
                children = payload.get("data", {}).get("children", [])
                for child in children:
                    kind = child.get("kind") or ""
                    data = child.get("data") or {}
                    name = data.get("name") or ""
                    if not name:
                        continue
                    out[name] = thing_text(kind, data)
                return out
            if resp.status_code == 429:
                print(
                    f"429 on batch ({len(ids)} ids), attempt {attempt}/{MAX_RETRIES}. "
                    f"Sleeping {delay}s."
                )
                time.sleep(delay)
                delay *= BACKOFF_FACTOR
            else:
                print(f"Batch fetch failed: HTTP {resp.status_code}")
                break
        except Exception as e:
            print(f"Error fetching batch (attempt {attempt}): {e}")
            time.sleep(delay)
            delay *= BACKOFF_FACTOR
    return {}


def expand_input_paths(paths: List[str]) -> List[str]:
    """Expand directory entries into their contained ``.ndjson`` files."""
    out: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            for fname in sorted(os.listdir(p)):
                full = os.path.join(p, fname)
                if os.path.isfile(full) and fname.endswith(".ndjson"):
                    out.append(full)
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f"Warning: input path not found, skipping: {p}")
    return out


def load_rows(paths: List[str]) -> List[Dict[str, str]]:
    """Read every line of every NDJSON file into a flat list of dicts.

    Only pulls the fields we need; silently skips malformed lines.
    """
    rows: List[Dict[str, str]] = []
    for path in paths:
        n_lines = 0
        n_bad = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                n_lines += 1
                rows.append(
                    {
                        "comment_id": as_clean_str(obj.get("comment_id")),
                        "comment_text": as_clean_str(obj.get("comment_text")),
                        "parent_id": as_clean_str(obj.get("parent_id")),
                        "_source": os.path.basename(path),
                    }
                )
        print(
            f"Loaded {n_lines:,} rows from {path}"
            + (f"  ({n_bad} malformed lines skipped)" if n_bad else "")
        )
    return rows


def main() -> None:
    input_files = expand_input_paths(INPUT_PATHS)
    if not input_files:
        raise SystemExit("No NDJSON input files found; check INPUT_PATHS.")

    print(f"Input files ({len(input_files)}):")
    for f in input_files:
        print(f"  {f}")

    rows = load_rows(input_files)
    print(f"Total rows loaded: {len(rows):,}")

    needed: set[str] = set()
    for row in rows:
        pid = normalize_fullname(row["parent_id"])
        if pid and (pid.startswith("t1_") or pid.startswith("t3_")):
            needed.add(pid)

    unique_ids = sorted(needed)
    print(f"Unique parent ids to fetch: {len(unique_ids):,}")

    id_to_text: Dict[str, str] = {}
    batch_index = 0
    total_batches = (len(unique_ids) + MAX_IDS_PER_REQUEST - 1) // MAX_IDS_PER_REQUEST

    for batch in chunks(unique_ids, MAX_IDS_PER_REQUEST):
        batch_index += 1
        gotten = fetch_info_batch(batch)
        id_to_text.update(gotten)
        missing = set(batch) - set(gotten.keys())
        if missing and len(missing) <= 10:
            print(f"  batch {batch_index}/{total_batches}: not returned: {missing}")
        elif missing:
            print(
                f"  batch {batch_index}/{total_batches}: "
                f"{len(missing)} ids not returned (deleted or missing)"
            )

        time.sleep(BATCH_DELAY)
        if batch_index % LONG_DELAY_EVERY == 0:
            print(f"{batch_index} batches done: sleeping {LONG_DELAY}s...")
            time.sleep(LONG_DELAY)

    out_rows: List[Dict[str, str]] = []
    for row in rows:
        cid = row["comment_id"]
        comment_text = row["comment_text"] or ""
        pid = row["parent_id"] or ""
        out_rows.append(
            {
                "comment_id": cid,
                "comment_text": comment_text,
                "parent_id": pid,
                "parent_text": id_to_text.get(pid, ""),
            }
        )

    out_dir = os.path.dirname(os.path.abspath(OUTPUT_CSV))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(out_rows, columns=["comment_id", "comment_text", "parent_id", "parent_text"])
    df_out.to_csv(OUTPUT_CSV, index=False)

    n_empty_parent = int((df_out["parent_text"] == "").sum())
    print(
        f"Done! Wrote {len(out_rows):,} rows to {OUTPUT_CSV}\n"
        f"  parent_text empty : {n_empty_parent:,} "
        f"({100.0 * n_empty_parent / max(len(out_rows), 1):.1f}%)"
    )


if __name__ == "__main__":
    main()
