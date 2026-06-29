# In Tandem — Product Data Scientist take-home

**Allocate a retention budget across offers.** For ~40,000 subscribers, decide which
offer (none / $1 nudge / $5 discount / $15 concierge) to give each — within a
**$40,000 budget** — to maximize **net incremental retained revenue**. The catch:
the best offer differs per user, bigger offers backfire on some ("sleeping dogs"),
and one feature in the data is a measurement trap.

👉 **Read [`TASK.md`](TASK.md) first** — full brief, what to submit, how you're evaluated.

## Quick start
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# data is in data/ — model on data/train.csv (has offer_arm, churned),
# validate on data/holdout.csv, allocate offers for data/scoring.csv.
# READ data/data_dictionary.md — one column is a trap.
```
Or in Colab: paste `colab_bootstrap.txt` into the first cell.

## What you submit
1. `allocation.csv` — `user_id,offer_arm` (0–3) for the scoring users, total cost ≤ $40,000.
2. `holdout_scores.csv` — `user_id,uplift_score` for **every** `holdout.csv` row.
3. Your code **with the development visible** (notebook with intermediate steps, or repo
   with commits) — not just a final clean cell.
4. A ~1–2 page writeup **including an iteration log**: the versions you tried, what each
   scored, and why you changed approach. **Showing your development process is part of
   the task** — we weight it as much as the final number.

Deliver as a **GitHub repo** (preferred) or a single runnable **Colab notebook**.

## Notes
- Data is **synthetic** — no real user data.
- Time: **open-ended** — no time limit, and you're not expected to finish everything.
  Submit what you have and show your working.
- AI assistance is encouraged — just note how you used it. (A one-shot with no visible
  iteration scores poorly — see TASK.md §7.)
