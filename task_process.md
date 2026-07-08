---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
!pip -q install -r requirements.txt
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
train   = pd.read_csv("data/train.csv")     # randomized experiment: offer_arm (0-3), churned
holdout = pd.read_csv("data/holdout.csv")   # disjoint slice for HONEST evaluation (don't train on it)
scoring = pd.read_csv("data/scoring.csv")   # prospective users to allocate offers for (no offer_arm/churned)
print(train.shape, holdout.shape, scoring.shape)
```

# 1.EDA


In this section I check general info about data to get to know it better. I will answer the following questions:
1. Are there NaNs?
2. What type every feature has?
3. How many unique values every feature has?
4. Do three datasets have the same nature?  - here i'll check distributions/descriptive statistics 


**Main outcomes of the section**

1. There are no NaN values, therefore no need to clean;
2. Acquisition_channel is represented by text, threfore it should be encoded if used further and model doesn't handle categorical variables natively
3. For features autopay, acquisition_channel, price_tier,offer_arm, churn are categorical variables. They should not be treated as continuous numerical features.
4. The feature distributions are generally consistent across the train, holdout, and scoring datasets, suggesting they come from the same underlying population. The only exception is offer_window_logins, whose distribution differs because it is a placeholder in the scoring dataset and should not be used as a predictive feature due to the risk of data leakage.
5. Annual_value = 12 × mrr. No need to use both if to include to prediction

```python
print(train.info())
print(holdout.info())
print(scoring.info())
```

```python
check_unique_values = pd.concat(
    [
        train.nunique().rename("train"),
        holdout.nunique().rename("holdout"),
        scoring.nunique().rename("scoring"),
    ],
    axis=1,
)

print(check_unique_values)
```

```python
datasets = {
    "train": train,
    "holdout": holdout,
    "scoring": scoring
}


for col in train.columns[1:]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Distribution of {col}", fontsize=14)

    for ax, (name, df) in zip(axes, datasets.items()):
        if col not in df.columns:
            ax.set_title(f"{name}: missing column")
            ax.axis("off")
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(name)
            ax.set_xlabel(col)
        else:
            value_counts = df[col].value_counts(dropna=False).head(20)
            sns.barplot(
                x=value_counts.values,
                y=value_counts.index.astype(str),
                ax=ax
            )
            ax.set_title(name)
            ax.set_xlabel("count")
            ax.set_ylabel(col)

    plt.tight_layout()
    plt.show()
```

# 2. Churn investigation + Sleeping dogs


On this step  I will explore the main dependencies beyween churn and features. I will focus on the next questions:
1. Whether offers are randomely distributes, what's proportion of each one
2. How/If offer_arm and churn are dependent
3. Explore features across offer arm and churn in order to find out who each offer helps vs. hurts and find sleeping dogs

```python
train.columns
```

**Main outcomes of the section**

1. Offer arms are randomly distributed with approximately equal proportions, confirming successful randomization.
2. Overall churn rates are similar across the train and holdout datasets, suggesting both datasets come from the same distribution.
3. Segment-level analysis reveals heterogeneous treatment effects that are hidden by overall averages.
- tenure_months – Offers appear to reduce churn primarily for recently acquired customers. For customers with long tenure, some offers are associated with slightly higher churn rates, suggesting the presence of potential sleeping dogs.
- active_days_30d – Customers with very low activity (0–1 active days) exhibit higher churn when contacted, indicating a possible sleeping-dog segment. Customers active for 2–4 days show the largest reduction in churn after receiving an offer, whereas highly active customers experience little or no measurable benefit.
- sessions_30d – Customers with very few sessions show little response to any offer, suggesting limited incremental value from targeting this segment.
- family_members – Customers with four family members exhibit approximately a 3 percentage point reduction in churn after receiving an offer, indicating a segment that responds well to retention campaigns.
- features_used – Customers using one or two product features experience a 2–3 percentage point reduction in churn under treatment, whereas heavier users show smaller differences across offer arms.
- support_tickets_90d – Customers with no recent support interactions show little response to offers. Customers with many support tickets appear to churn more frequently after receiving an offer, suggesting they may represent another sleeping-dog segment. This pattern should be validated by the uplift model.
- price_tier – Offers reduce churn by less than approximately 2 percentage points across all price tiers. No single price tier appears to respond substantially more than the others.
- months_since_last_active – No clear treatment effect is observed across activity recency groups.
- autopay – Customers enrolled in autopay show little change in churn regardless of the offer received. In contrast, customers without autopay respond positively to retention offers, with larger offers generally producing larger reductions in churn.
- prior_offers – Customers who previously received more than four offers appear to benefit substantially from additional offers. Conversely, customers with two to three prior offers show increased churn under treatment, suggesting a possible sleeping-dog group that warrants further investigation.
- acquisition_channel – Customers acquired through the App Store respond positively to all offers, while customers acquired via Paid Search appear to benefit primarily from offer arm 2.
- mrr – No meaningful differences in treatment effect are observed across monthly recurring revenue levels.

```python
# Offer_arm VS Churn

print(f'Offer arm proportions train:\n {train["offer_arm"].value_counts(normalize=True)}')
print(f'Offer arm proportions holdout:\n {holdout["offer_arm"].value_counts(normalize=True)}')

arm_vs_churn_train = pd.crosstab(train["offer_arm"],train["churned"],normalize="index"
) * 100
arm_vs_churn_holdout = pd.crosstab(holdout["offer_arm"],holdout["churned"],normalize="index"
) * 100
print(f'Churn rate by offer train:\n {arm_vs_churn_train}')
print(f'Churn rate by offer holdout:\n {arm_vs_churn_holdout}')

```

```python
# Explore features across Churn and Offer arm
```

```python

def explore_churn_rates(df, feature, numerical):
    if numerical:
        df["months_since_last_active_bins"] = pd.qcut(df["months_since_last_active"], q=3)
        churn_rate = df.groupby([f'{feature}_bins', "offer_arm"])["churned"].mean().unstack()
    else:
        churn_rate = df.groupby([feature, "offer_arm"])["churned"].mean().unstack()

    # Difference in percentage points relative to the first column
    result = churn_rate.copy()
    baseline = churn_rate.iloc[:, 0]

    for col in churn_rate.columns[1:]:
        result[f"{col} vs {churn_rate.columns[0]} (pp)"] = (
            (churn_rate[col] - baseline) * 100
        ).round(2)

    print(result)
```

```python
train_df = train.copy()
```

```python
# Price tier
explore_churn_rates(train_df, "price_tier", False)
```

```python
# Autopay
explore_churn_rates(train_df, "autopay", False)

```

```python
# Active days last month
explore_churn_rates(train_df, "active_days_30d", False)
```

```python
# Family members
explore_churn_rates(train_df, "family_members", False)
```

```python
# features used
explore_churn_rates(train_df, "features_used", False)
```

```python
# support tickets
explore_churn_rates(train_df, "support_tickets_90d", False)
```

```python
#prior offers
explore_churn_rates(train_df, "prior_offers", False)
```

```python
# acquisition channel
explore_churn_rates(train_df, "acquisition_channel", False)
```

```python
# Tenure months
explore_churn_rates(train_df, "tenure_months", True)
```

```python
# sessions
explore_churn_rates(train_df, "sessions_30d", True)
```

```python
# months since last active
explore_churn_rates(train_df, "months_since_last_active", True)
```

```python
# MRR
explore_churn_rates(train_df, "mrr", True)
```

<!-- #region -->
# Modeling

**Main outcomes of the section**

To establish a naive baseline, I first built a churn prediction model using XGBoost, which is a powerful gradient boosting algorithm widely used for binary classification tasks. As a preprocessing step, the categorical feature acquisition_channel was one-hot encoded into dummy variables.

The model achieved a ROC AUC of 0.63, indicating a moderate ability to distinguish between customers who churn and those who remain. The PR AUC was 0.50, which is substantially higher than the baseline churn rate of approximately 38%, suggesting that the model identifies high-risk customers better than random guessing.

Following the naive strategy described in the assignment, I ranked customers by their predicted churn probability and assigned the most expensive retention offer (arm 3, concierge) to the highest-risk users, while all remaining customers received no offer (arm 0).
Only observations where the randomized offer matched the assigned offer contributed to the estimate. Since only about one-quarter of customers matched, the results were scaled accordingly to estimate the value for the full population.

Compared with the "no-offer" scenario, the naive churn-based strategy produced an incremental net value of approximately −$42.9k. This negative result demonstrates that customers with the highest predicted churn risk are not necessarily the customers who are most responsive to retention offers.



As first uplift modeling approach I used a T-learner. I trained one separate churn model for each randomized offer arm and then predicted each user’s potential churn probability under every arm. Because the offer was randomized, differences between the predicted churn probability under no offer and under each paid offer can be interpreted as estimated per-user treatment effects. This approach is simple, transparent, and well suited as a first causal baseline after the naive churn model.
<!-- #endregion -->

```python
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)
from sklearn.metrics import classification_report
```

```python
BUDGET = 40000
COST_ARM_MAP = { 0:0,
                1:1,
                2:5,
                3:15
                }
```

```python
# Encode acquisition channel

train = pd.get_dummies(train,
    columns=["acquisition_channel"],
    drop_first=False,
    dtype=int
)

holdout = pd.get_dummies(holdout,
    columns=["acquisition_channel"],
    drop_first=False,
    dtype=int
)

scoring = pd.get_dummies(scoring,
    columns=["acquisition_channel"],
    drop_first=False,
    dtype=int
)

```

```python
train_features = ['tenure_months', 'active_days_30d', 'sessions_30d',
       'family_members', 'features_used', 'support_tickets_90d', 'price_tier',
       'months_since_last_active', 'autopay', 'prior_offers', 'mrr',  
       'acquisition_channel_app_store', 'acquisition_channel_organic',
       'acquisition_channel_paid_search', 'acquisition_channel_referral']
```

```python
X_train = train[train_features]
y_train = train['churned']

X_test = holdout[train_features]
y_test = holdout['churned']

X_score = scoring[train_features]
```

## 3. Baseline churn model

```python
train["churned"].value_counts(normalize=True)
```

```python
model_baseline = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model_baseline.fit(X_train, y_train)
y_pred_proba = model_baseline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)
```

```python
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("PR AUC :", average_precision_score(y_test, y_pred_proba))
```

```python
print(classification_report(y_test, y_pred))
```

```python
# Check value of the baseline model
holdout["baseline_pred_proba"] = y_pred_proba 

n_target = BUDGET //COST_ARM_MAP[3]
baseline_cost = n_target*COST_ARM_MAP[3]
holdout['baseline_offer_arm'] = 0

high_risk = (
    holdout
    .sort_values("baseline_pred_proba", ascending=False)
    .head(n_target)
    .index
)


holdout.loc[high_risk, "baseline_offer_arm"] = 3
```

```python
matched_offer_arm = holdout[holdout['offer_arm'] == holdout['baseline_offer_arm']].copy()
```

```python
# Value of model with offer 3 for high risk users 
matched_offer_arm["retained_value"] = ((1 - matched_offer_arm["churned"])  * matched_offer_arm["annual_value"])
matched_offer_arm["offer_cost"] = matched_offer_arm["baseline_offer_arm"].map(COST_ARM_MAP)
matched_offer_arm["net_value"] = ( matched_offer_arm["retained_value"] - matched_offer_arm["offer_cost"])
net_baseline = matched_offer_arm["net_value"].sum()
print(f"Matched rows:{matched_offer_arm.shape[0]}")
print(f'Net value of baseline: {net_baseline*4}')

# Value of model without offer
control = holdout[holdout["offer_arm"] == 0].copy()
control["net_value"] = ((1 - control["churned"]) * control["annual_value"])
control_net_value = control["net_value"].sum()
print(f"Control matched rows: {control.shape[0]}")
print(f'Net value of control: {control_net_value*4}')

# incremental effetc of baseline model
incremental_value_baseline = net_baseline*4 - control_net_value*4
print(f'Incremental value of baseline: {incremental_value_baseline}')

```

## 4. Effect of each offer per User

```python
t_learner_models = {}
arms = [0, 1, 2, 3]

for arm in arms:
    train_arm = train[train["offer_arm"] == arm].copy()

    X_arm = train_arm[train_features]
    y_arm = train_arm["churned"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_arm, y_arm)

    t_learner_models[arm] = model

    # This AUC is only diagnostic, not the final success metric
    holdout_pred = model.predict_proba(holdout[train_features])[:, 1]

    print(f"Arm {arm}")
    print("Rows:", train_arm.shape[0])
    print("Holdout ROC AUC:", roc_auc_score(holdout["churned"], holdout_pred))
    print("Holdout PR AUC :", average_precision_score(holdout["churned"], holdout_pred))
    print('-'*50)
```

```python
for arm in arms:
    holdout[f"t_pred_churn_arm_{arm}"] = t_learner_models[arm].predict_proba( holdout[train_features] )[:, 1]

for arm in [1, 2, 3]:
    holdout[f"t_uplift_arm_{arm}"] = (holdout["t_pred_churn_arm_0"] - holdout[f"t_pred_churn_arm_{arm}"] )

for arm in [1, 2, 3]:
    holdout[f"t_expected_net_value_arm_{arm}"] = (holdout[f"t_uplift_arm_{arm}"] * holdout["annual_value"] - COST_ARM_MAP[arm] )
```

```python
uplift_cols = [
    "t_uplift_arm_1",
    "t_uplift_arm_2",
    "t_uplift_arm_3"
]

net_value_cols = [
    "t_expected_net_value_arm_1",
    "t_expected_net_value_arm_2",
    "t_expected_net_value_arm_3"
]

print("Uplift summary")
holdout[uplift_cols].describe()
```

```python
print("Expected net value summary")
holdout[net_value_cols].describe()
```

```python
candidate_value_cols = {
    1: "t_expected_net_value_arm_1",
    2: "t_expected_net_value_arm_2",
    3: "t_expected_net_value_arm_3"
}

holdout["t_best_arm"] = 0
holdout["t_best_value"] = 0.0

for arm, col in candidate_value_cols.items():
    better = holdout[col] > holdout["t_best_value"]
    holdout.loc[better, "t_best_arm"] = arm
    holdout.loc[better, "t_best_value"] = holdout.loc[better, col]

holdout["t_best_arm"].value_counts(normalize=True)
```

```python
holdout[["offer_arm", "t_best_arm","churned"]]
```

```python

```
