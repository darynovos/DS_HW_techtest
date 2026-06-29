# Data dictionary — In Tandem retention-offer experiment

One row per subscriber. `train.csv` and `holdout.csv` are a **randomized
experiment**: the retention offer (`treatment`) was assigned by a fair coin, so
it is independent of all features. `scoring.csv` is a separate set of prospective
subscribers with **no** `treatment`/`churned` — your job is to decide who to treat.

| column | type | meaning |
|---|---|---|
| user_id | int | unique subscriber id |
| tenure_months | float | months subscribed |
| active_days_30d | int 0–7 | active days in the last week-month |
| sessions_30d | int | app sessions last 30 days |
| family_members | int 1–6 | household size on the plan |
| features_used | int 1–8 | breadth of app features used |
| support_tickets_90d | int | support contacts last 90 days |
| price_tier | int 0/1/2 | basic / plus / premium |
| months_since_last_active | float | recency of last activity |
| autopay | 0/1 | autopay enabled |
| prior_offers | int | retention offers already received |
| acquisition_channel | str | organic / paid_search / referral / app_store |
| mrr | float $ | monthly recurring revenue from this subscriber |
| annual_value | float $ | 12 × mrr — the value lost if they churn (use for ROI) |
| treatment | 0/1 | **(train/holdout only)** received the retention offer |
| churned | 0/1 | **(train/holdout only)** churned in the outcome window |

**Economics (fixed):** each treatment costs **$5.00**. A retained subscriber is
worth their `annual_value`. Your budget targets **15% of the scoring set**.
