"""Fetch Reddit parent and submission context text for annotated rows."""

import time
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests

# === CONFIGURE HERE ===
INPUT_CSV = "./Data/Dan annotations5k 04202022/fetched_data.csv"
OUTPUT_CSV = "./Data/Dan annotations5k 04202022/fetched_context_data.csv"

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


def normalize_fullname(s: Optional[str]) -> Optional[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    t = str(s).strip()
    return t if t else None


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


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    needed: set[str] = set()
    for col in ("parent_id", "parent_post_id"):
        for v in df[col].tolist():
            fn = normalize_fullname(v)
            if fn and (fn.startswith("t1_") or fn.startswith("t3_")):
                needed.add(fn)

    unique_ids = sorted(needed)
    print(f"Unique Reddit things to fetch: {len(unique_ids)}")

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

    rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        cid = normalize_fullname(row.get("comment_id")) or ""
        body = row.get("body")
        comment_text = "" if body is None or (isinstance(body, float) and pd.isna(body)) else str(body)

        pid = normalize_fullname(row.get("parent_id")) or ""
        ppid = normalize_fullname(row.get("parent_post_id")) or ""

        rows.append(
            {
                "comment_id": cid,
                "comment_text": comment_text,
                "parent_id": pid,
                "parent_text": id_to_text.get(pid, ""),
                "parent_post_id": ppid,
                "parent_post_text": id_to_text.get(ppid, ""),
            }
        )

    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
