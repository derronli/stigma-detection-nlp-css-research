from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

if torch.cuda.is_available():
    device = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True)
model = AutoModel.from_pretrained("vinai/bertweet-base").to(device)
model.eval()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "Data/Dan annotations5k 04202022/final.csv"
OUTPUT_ROOT = PROJECT_ROOT / "Data/embeddings"
CONSOLIDATED_EMBEDDINGS = OUTPUT_ROOT / "embeddings.npy"
CONSOLIDATED_ALIGNMENT_KEYS = OUTPUT_ROOT / "comment_ids.npy"
SAVE_PER_COMMENT_FOLDERS = False
TEXT_COL = "body"
ID_COL = "comment_id"
MAX_LENGTH = 128
BATCH_SIZE = 32


def get_berttweet_embeddings(texts):
    """Return CLS embeddings for a batch of texts as np.ndarray [batch, hidden]."""
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


def save_embedding(comment_id, embedding):
    """Save one embedding as .npy under data/embeddings/{comment_id}/embedding.npy."""
    target_dir = OUTPUT_ROOT / str(comment_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    np.save(target_dir / "embedding.npy", embedding)


def main():
    df = pd.read_csv(INPUT_CSV)
    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing required text column: '{TEXT_COL}'")
    if ID_COL not in df.columns:
        raise ValueError(f"Missing required id column: '{ID_COL}'")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = len(df)
    hidden = model.config.hidden_size
    all_embeddings = np.zeros((total, hidden), dtype=np.float32)
    all_ids = np.empty(total, dtype=object)

    for start in tqdm(range(0, total, BATCH_SIZE), desc="Encoding comments"):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]
        batch_emb = get_berttweet_embeddings(batch[TEXT_COL].tolist())
        all_embeddings[start:end] = batch_emb
        all_ids[start:end] = batch[ID_COL].astype(object).to_numpy()

        if SAVE_PER_COMMENT_FOLDERS:
            for comment_id, emb in zip(batch[ID_COL].tolist(), batch_emb):
                save_embedding(comment_id, emb)

    np.save(CONSOLIDATED_EMBEDDINGS, all_embeddings)
    np.save(CONSOLIDATED_ALIGNMENT_KEYS, all_ids)

    print(
        f"Done! Saved consolidated arrays for {total} rows:\n"
        f"  - {CONSOLIDATED_EMBEDDINGS}  shape {all_embeddings.shape}\n"
        f"  - {CONSOLIDATED_ALIGNMENT_KEYS}  (join keys only; same row order as INPUT_CSV)"
    )
    if SAVE_PER_COMMENT_FOLDERS:
        print(f"  Also saved per-id files under '{OUTPUT_ROOT}/{{comment_id}}/embedding.npy'.")


if __name__ == "__main__":
    main()