# In Tandem — Product Data Scientist take-home

**Causal targeting under a retention budget.** Decide which **15%** of subscribers
to send a retention offer to, to maximize **net incremental retained revenue** —
telling causal *uplift* apart from raw *churn risk*.

👉 **Read [`TASK.md`](TASK.md) first** — it's the full brief, what to submit, and how you're evaluated.

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# data is already in data/ — start modeling on data/train.csv, validate on data/holdout.csv,
# decide who to treat in data/scoring.csv. See data/data_dictionary.md for columns.
```
Or in Colab: paste `colab_bootstrap.txt` into the first cell.

## What you submit
1. `target.csv` — `user_id` of the ≤6,000 subscribers you'd treat (from `scoring.csv`).
2. `holdout_scores.csv` — `user_id,uplift_score` for **every** `holdout.csv` row.
3. Your reproducible code (this repo, or a Colab notebook) + a ~1-page writeup.

Deliver as a **GitHub repo** (preferred) or a single runnable **Colab notebook**.

## Notes
- Data is **synthetic** — no real user data.
- Time: aim for **3–4 hours**; submit what you have and say where you scoped.
- AI assistance is encouraged — just note how you used it.
