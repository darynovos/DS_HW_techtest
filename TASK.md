# Data Scientist Screening Task: Allocating a Retention Budget Across Offers

**Role:** Product Data Scientist, In Tandem
**Stage:** Technical exercise (for candidates who cleared the résumé/application screen)
**Format:** Build-anywhere take-home on a **free environment** — Google Colab, Kaggle, or Databricks Community Edition. We provide a synthetic dataset (no proprietary data, no paid compute). Self-reported results; we re-run the top submissions to confirm they reproduce.
**Time guidance:** Open-ended — spend as much or as little as you like. We're not timing you, and we don't expect every part finished. **Submit what you have, scope honestly, and show your working** (see §5).

---

## 1. What the task is

A model that predicts **who will churn** is the easy version. The real job is deciding
**how to spend a limited retention budget across several possible offers** so the
business keeps the most revenue. Predicting churn and allocating a budget are *not*
the same problem, and the people you'd target differ.

We ran a retention experiment on a consumer subscription app. Each user was randomly
given one of **four offers**:

| arm | offer | cost |
|---|---|---|
| 0 | none | $0 |
| 1 | nudge (in-app message) | $1 |
| 2 | discount | $5 |
| 3 | concierge (human outreach) | $15 |

> Your job: for a fresh set of ~40,000 prospective subscribers, **decide which offer
> (if any) to give each one**, spending at most a **$40,000 total budget**, to
> **maximize net incremental retained revenue** = value saved by retaining them −
> the cost of the offers you sent.

There is no single right answer. We measure whether you can tell **causal impact from
correlation**, **allocate a budget across options**, handle **messy data**, and
**defend the decision** — not whether you hit one accuracy number.

> **Why it's hard.** Bigger offers help *persuadable* users more but cost more, and a
> bigger offer **annoys some users into leaving faster** (negative effect — "sleeping
> dogs"). Crucially, the *best* offer differs per user: some are won over by a cheap
> nudge (concierge is wasted on them); others only respond to concierge. And the
> budget won't stretch to give everyone the big offer. Targeting the highest-churn
> users with the biggest offer **destroys value**.

---

## 2. The data (provided — do not go data-shopping)

Three files in `data/` (already generated). Full column docs in `data/data_dictionary.md`.

| file | rows | what it is |
|---|---|---|
| `train.csv` | ~110k | **Randomized experiment.** `offer_arm` (0–3) was assigned at random → independent of features. Has the outcome `churned`. |
| `holdout.csv` | ~50k | Same schema — a disjoint slice for **honest** evaluation (don't train on it). |
| `scoring.csv` | ~40k | Prospective subscribers — **no `offer_arm`, no `churned`.** You assign each an offer. |

**⚠ Read the data dictionary carefully** — one feature is a measurement trap. Part of
the job is spotting which fields you can legitimately use to target *prospective*
users, and which would leak.

**Economics (fixed):** arm costs as above; a retained subscriber is worth their
`annual_value` (= 12 × MRR); **total budget $40,000** across the scoring set.
Because offers were randomized, **per-arm uplift / CATE is identifiable directly** —
this is a causal-allocation exercise, not a confounding-adjustment one.

---

## 3. How to approach it

1. **Look before you model.** How does churn differ by arm overall? (The averages are
   small — they hide who each offer helps vs. hurts.)
2. **Baseline, honestly.** Build the naive version — predict churn, give high-risk
   users the big offer — and measure its net value. It's the floor you must beat.
3. **Estimate the effect of *each* offer per user** (per-arm uplift / CATE — T-/S-/
   X-learner, uplift trees, causal forest; your choice, justify it).
4. **Allocate under the budget.** For each user pick the arm maximizing
   `uplift(arm) × annual_value − cost(arm)`; spend the $40k where each dollar earns
   most (think value-per-dollar, not just value). Don't treat users the offer hurts.
5. **Validate on the holdout** — uplift curve / Qini, and the headline: **net
   incremental value of your allocation vs the churn baseline.**
6. **Write it up + show your working** (§5).

---

## 4. Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, scikit-learn (pinned)
# add any uplift library you like and PIN it: scikit-uplift / causalml / econml
```
Data is already in `data/`. Use it as given — **do not modify or regenerate the CSVs.**
A one-cell Colab bootstrap is in `colab_bootstrap.txt`.

---

## 5. What to submit

1. **Your allocation:** `allocation.csv` with columns `user_id,offer_arm` (0–3) for the
   scoring users, total cost ≤ $40,000.
2. **Your holdout uplift scores:** `holdout_scores.csv` with `user_id,uplift_score`
   over **every** `holdout.csv` row (so we can compute your Qini).
3. **Your code, with the development visible** — a notebook (or repo) we can run from
   clean. **Don't hand us only the final clean cell.** We want to see how you got
   there.
4. **A short writeup (~1–2 pages)** — and an explicit **iteration log**:

   > **Showing your development process is part of the task, not a formality.** We
   > care more about *how you reasoned and improved* than the final number. Include,
   > in your own words:
   > - **Iteration log:** the versions you went through and what each scored —
   >   e.g. "v1 churn model → measured net value → realized I was funding sleeping
   >   dogs → v2 per-arm uplift → v3 budget allocation by value-per-dollar." Show the
   >   numbers at each step (they should match what our grader recomputes from your code).
   > - **What you found in the data** (including the measurement trap) and what you did about it.
   > - **The headline:** net value of your allocation vs the churn baseline; your Qini;
   >   and what allocating across offers changed vs a single-offer policy.
   > - **Sleeping dogs:** how you found the negative-effect group and kept offers away from them.
   > - **The budget call:** how you'd reallocate if the budget were halved.
   > - **Production transfer:** how you'd ship this on **Databricks + AWS** (feature
   >   pipeline, refresh cadence, monitoring effect decay, retraining triggers) and how
   >   you'd explain the plan to **Finance (ROI)** and **Product (which offer to ship)**.

   A one-shot final artifact with no visible iteration scores poorly even if the number
   is decent — we're hiring for judgment under ambiguity, and judgment shows in the path.

---

## 6. How it is evaluated

We grade the **decision and the reasoning**, not a leaderboard. We score your
`allocation.csv` against held-back ground truth (the answer key is **not** in this
bundle — your job is to estimate it) and report **net incremental value**, the
**perfect-foresight optimum**, a **value-greedy baseline**, your **value capture %**,
your **holdout Qini**, your **arm mix**, and **how many sleeping dogs you paid to lose**.

Reference points on this dataset (Qini **normalized**: perfect ranking = 1.0, random = 0.0):

| Policy | Value capture | Qini |
|---|---|---|
| Value-greedy baseline | 0% | — |
| **Naive churn → big offer (the trap)** | **≈ −2%** | **≈ −0.03** |
| Competent uplift + budget allocation (the bar to beat) | ≈ 26% | ≈ 0.24 |
| Perfect foresight (ceiling) | 100% | 1.00 |

The naive policy is **worse than the baseline** — it funds lost causes and sleeping
dogs and over-spends on concierge. Persuadability and best-offer are genuinely hard,
non-linear signals, so even a strong solution captures a *minority* of the ceiling —
that's expected. A serious submission clears the baseline by a wide margin (≈20%+
capture), keeps offers off the sleeping dogs, and — critically — **shows the path it
took to get there.** This separates three groups:

- **Only predicted churn / gave high-risk users the big offer / no causal step / no iteration shown.** → out
- **Modeled uplift but** allocated naively (one offer for all, ignored value-per-dollar), used the leaky feature, never found the sleeping dogs, or **submitted a clean artifact with no visible development.** → borderline
- **Per-arm effects estimated; budget allocated by value-per-dollar; messy-data trap caught; sleeping dogs avoided; honest holdout validation; and a clear iteration log + business framing.** → advances

**We grade the method, not the exact dollar figure.** We re-run the top slice on the
provided data to confirm it reproduces and that the iteration log's numbers match.

---

## 7. Ground rules

- **AI assistance is encouraged** — note how you used it. But a one-shot-generated
  solution is easy to spot: it has no iteration log, includes the leaky feature, pays
  the sleeping dogs, and the writeup can't explain its own numbers. **Showing your
  development beats a polished one-shot.**
- **Don't fabricate.** Every number must reproduce from your code on the provided data.
  An iteration log whose numbers don't reconcile with a re-run is a strong negative.
- **Scope down honestly.** A clean baseline + per-arm uplift + an honest allocation,
  with the production section thin, beats a sprawling artifact you can't defend. If you
  ran out of time, say where — and show the steps you *did* take.

We're testing what a churn-accuracy task can't: **allocating a constrained budget across
options under messy, causal uncertainty — and reasoning through it in the open.** Good luck.
