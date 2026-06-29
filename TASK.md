# Data Scientist Screening Task: Causal Targeting Under a Retention Budget

**Role:** Product Data Scientist, In Tandem
**Stage:** Technical exercise (for candidates who cleared the résumé/application screen)
**Format:** Build-anywhere take-home on a **free environment** — Google Colab, Kaggle, or Databricks Community Edition. We provide a synthetic dataset (no proprietary data, no paid compute). Self-reported results; we re-run the top submissions to confirm they reproduce.
**Time guidance:** Aim for **3–4 hours.** Submit what you have.

---

## 1. What the task is

A retention model that predicts **who will churn** is the easy version. The real job — and what this task measures — is deciding **who to spend a limited retention budget on so the business keeps the most revenue.** Those are *not the same users.*

We ran a retention-offer experiment on a consumer subscription app. You get the experiment data. Your job:

> Choose which **15%** of a fresh set of subscribers should receive the retention offer, to **maximize net incremental retained revenue** — the value you actually *save by treating them*, minus what the offer costs.

There is no single correct answer. We measure whether you can tell **causal impact from correlation**, target a budget on **incremental value**, and defend the decision to **Product and Finance** — not whether you hit a particular accuracy number.

> **Why this is the point.** Some subscribers will churn no matter what you do (lost causes). Some will stay regardless (sure things). Some are **persuadable** — they stay *only* if you reach them. And some are **sleeping dogs** — the offer *annoys them into leaving faster*. A model that targets the highest churn probability spends the budget on the first and last groups and **destroys value**. The whole task is separating incremental impact from raw churn risk.

---

## 2. The data (provided — do not go data-shopping)

Three files are provided in `data/` (already generated for you). Full column docs in `data/data_dictionary.md`.

| file | rows | what it is |
|---|---|---|
| `train.csv` | ~110k | **Randomized experiment.** `treatment` (got the offer) was assigned by a fair coin → independent of all features. Has the outcome `churned`. |
| `holdout.csv` | ~50k | Same schema + outcomes — a disjoint slice for **honest** model evaluation (don't train on it). |
| `scoring.csv` | ~40k | Prospective subscribers — **no `treatment`, no `churned`.** This is who you decide to target. |

**Economics (fixed — state them back):** each treatment costs **$5.00**; a retained subscriber is worth their `annual_value` column (= 12 × MRR); your budget treats **15% of `scoring.csv`** (~6,000 users). Because treatment is randomized, **uplift / CATE is identifiable directly** — this is a causal-targeting exercise, not a confounding-adjustment one.

---

## 3. How to approach it

1. **Look before you model.** What's the overall treated-vs-control churn difference? (It's small — that's the trap. The average hides who the offer helps and who it hurts.)
2. **Baseline, honestly.** Build the naive version — a churn-probability model, target the top 15% most-likely-to-churn — and evaluate its incremental value. You'll use it as the floor you must beat (the reference numbers are in §6).
3. **Estimate uplift / CATE.** Model the *incremental* effect of the offer per subscriber (T-/S-/X-learner, uplift tree, causal forest — your choice, justify it). Rank by **expected net value** = `uplift × annual_value − cost`, not by uplift alone.
4. **Decide who to treat.** Pick the budget-sized target list. Decide what to do about users whose expected value is negative even though budget remains (hint: don't treat them).
5. **Measure incrementality on the holdout.** Report a **Qini / uplift curve** and the headline number: **net incremental retained revenue of your policy vs the churn-baseline policy.** Show how many users the two policies *disagree* on, and that your policy avoids the sleeping dogs.
6. **Write it up** (§5). Short, tied to your own numbers.

---

## 4. Reproduce the setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # numpy, pandas, scikit-learn (pinned)
# add any uplift library you like and PIN it: scikit-uplift / causalml / econml
```
The data is **already provided** in `data/` (`train.csv`, `holdout.csv`,
`scoring.csv` + `data_dictionary.md`). Use it exactly as given — **do not modify or
regenerate the CSVs** — so submissions are comparable. A one-cell Colab bootstrap is
in `colab_bootstrap.txt`.

---

## 5. What to submit

1. **Your target list:** `target.csv` with a single `user_id` column — the ≤6,000 subscribers from `scoring.csv` you chose to treat.
2. **Your holdout uplift scores:** `holdout_scores.csv` with `user_id,uplift_score` over every `holdout.csv` row (so we can compute your Qini).
3. **Reproducible code** (notebook or repo) that runs from clean on the free tier and produces both files. Pin your dependencies. We must be able to re-run it from clean on the provided data.
4. **A short writeup (~1 page):**
   - Your uplift method and **why** you chose it over the churn-probability baseline.
   - **The headline:** net incremental retained revenue of your policy vs the churn baseline, and your Qini. What did targeting on *uplift* change vs targeting on *churn risk*?
   - **Sleeping dogs:** show you found the negative-uplift group and excluded them. How much value would the naive policy have destroyed?
   - The budget call: how deep you treated and what you'd do if the budget were cut in half.
   - **Production transfer:** how you'd ship this at In Tandem on **Databricks + AWS** — feature pipeline, refresh cadence, monitoring uplift decay, retraining triggers — and how you'd explain the target list to **Finance (ROI)** and **Product (which intervention to ship)**.

---

## 6. How it is evaluated

We grade the **decision and the reasoning**, not a leaderboard score. We score your
`target.csv` against the held-back ground truth and report **net incremental value**,
the **perfect-foresight optimum**, the **random floor**, your **value capture %**, your
**holdout Qini**, and **how many sleeping dogs you treated**. (The ground-truth answer
key is not included in this bundle — that's deliberate; your job is to estimate it.)

Reference points on this dataset (Qini is **normalized** so a perfect uplift ranking =
1.0 and random = 0.0):

| Policy | Net value | Value capture | Qini (norm.) |
|---|---|---|---|
| Random targeting (floor) | ≈ −$18.6k | 0% | 0.00 |
| **Churn top-N (the naive trap)** | **≈ −$22k** | **≈ −1.5%** | **≈ −0.09** |
| Competent uplift model (the bar to beat) | ≈ +$42k | ≈ 26% | ≈ 0.28 |
| Perfect foresight (ceiling) | ≈ +$214k | 100% | 1.00 |

Note the naive churn policy is **worse than random** — it concentrates lost causes and
sleeping dogs. A serious submission must land **clearly positive on value capture
(roughly 20%+) and beat the churn baseline**, with a positive Qini. Persuadability here
is a genuinely hard, non-linear signal, so even a strong uplift model captures a *minority*
of the perfect-foresight ceiling — that's expected; we reward beating the baseline and
honest measurement, not chasing 100%. This separates three groups:

- **Only built a churn model / targeted top-N churners / no causal evaluation / trained-on-holdout metrics.** → out
- **Built an uplift model but evaluated it wrong** (accuracy/AUC instead of incrementality, no Qini, never noticed the sleeping dogs, or treated negative-EV users to "use the budget"). → out
- **Two policies compared; incrementality measured honestly on the holdout; the predictive-vs-causal gap quantified in dollars; sleeping dogs found and excluded; budget decision defended; writeup that transfers to production and speaks to both Finance and Product.** → advances

**We grade the method, not the exact dollar figure.** We re-run the top slice from the
provided data to confirm the numbers reproduce. A generic "fit GBM on churn, take top 15%"
clears nothing; *measuring incremental value, finding the sleeping dogs, and defending
the budget call* is what separates candidates.

---

## 7. Ground rules

- **AI assistance is encouraged.** Note briefly how you used it — that's a plus. But a one-shot-generated churn model is obvious: it has a negative Qini, treats sleeping dogs, and the writeup can't explain why its "high-risk" targets lost money. We ship that exact baseline (§6) — clearing it is the floor, not the finish.
- **Don't fabricate.** Every number must come from a run we can reproduce on the provided data. Round numbers, a Qini with no curve, or "I targeted the highest-risk users" with no incremental measurement are tells.
- **Scope down honestly.** A clean uplift model + honest Qini + the sleeping-dog finding, with the production section thin, beats a sprawling submission you can't defend. If you only got the baseline-vs-uplift comparison done, submit that and say so.

We're testing the one thing a churn-accuracy task can't: **deciding where a limited budget creates the most incremental value, and defending it to the business.** Good luck.
