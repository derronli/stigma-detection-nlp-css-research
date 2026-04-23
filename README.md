# Context-Aware Stigma Detection Model: Comparing Insider vs. Outsider Stigmatizing Language Towards People Who Use Substances

## Files and usage

- `src/filter_reddit_drug_comments.py`: Streams `.zst` Reddit comment dumps, filters by date/keywords, and writes projected rows for downstream modeling. Command: `python src/filter_reddit_drug_comments.py`.
- `src/fetch_context_data.py`: Expands annotated rows with `parent_text` and `parent_post_text` via Reddit `/api/info` lookups. Command: `python src/fetch_context_data.py`.
- `src/fetch_parent_text.py`: Fetches only `parent_text` for filtered NDJSON comment files and writes a compact CSV. Command: `python src/fetch_parent_text.py`.
- `src/bert.py`: Encodes the `body` column into single-vector BERTweet embeddings aligned to `comment_id`. Command: `python src/bert.py`.
- `src/bert_multitext.py`: Encodes multiple text columns independently and concatenates embeddings into one multitext matrix. Command: `python src/bert_multitext.py --fields comment_text,parent_text`.
- `src/bert_pair.py`: Encodes `comment_text` and `parent_text` jointly as a text pair to produce one pooled vector per row. Command: `python src/bert_pair.py --output-dir Data/embeddings_multitext_pair`.
- `src/trainer.py`: Trains DCN/DCNv2 using single-text embeddings, demographics, and annotator IDs. Command: `python src/trainer.py --model dcn`.
- `src/trainer_multitext.py`: Trains DCN/DCNv2 on multitext embedding matrices and saves best checkpoints by validation loss. Command: `python src/trainer_multitext.py --model dcnv2`.
- `src/cv_multitext.py`: Runs comment-level K-fold cross-validation for multitext DCN/DCNv2 setups and reports fold metrics. Command: `python src/cv_multitext.py --n-splits 5 --model dcnv2`.
- `src/infer_multitext_pair.py`: Loads a trained multitext-pair checkpoint and writes jury-aggregated predictions for a new CSV. Command: `python src/infer_multitext_pair.py --input-csv input.csv --output-csv predictions.csv`.
- `src/data_split.py`: Provides comment-level split utilities used by training/evaluation code to avoid leakage across duplicated comments.
- `src/multitext/merge_final_context.py`: Left-joins `final.csv` with fetched context data and normalizes context text columns.
- `src/multitext/text_fields.py`: Defines allowed/default multitext fields and helper parsing for CLI field selection.
- `src/multitext/__init__.py`: Re-exports multitext helper symbols for cleaner imports in embedding scripts.
- `src/model/dcn.py`: Implements the baseline Deep & Cross Network used for stigma classification.
- `src/model/dcnv2.py`: Implements the DCNv2 variant with a matrix-style cross layer and the same training interface.
- `src/model/hierarchical_context.py`: Implements hierarchical context fusion and wrapper models for turn-level embedding inputs.
- `src/model/__init__.py`: Exposes model package exports (`DCN`, `DCNv2`, hierarchical classes) for shared imports. 

