
## 1. Data review 

I first checked missing values, feature types, cardinality, and distribution consistency across train, holdout, and scoring datasets.

The train and holdout datasets were broadly consistent, and offer arms were approximately evenly distributed, confirming that the randomized experiment was suitable for estimating treatment effects.

One important issue was the feature `offer_window_logins`. Its distribution differed in the scoring dataset, where it appeared to be a placeholder rather than an observed pre-treatment feature. Since this feature would not be available for prospective users at decision time, I excluded it from the modeling features to avoid leakage.

I also noted that `annual_value = 12 × mrr`, so I did not need to use both as independent modeling signals. I used `mrr` as a feature and used `annual_value` for economic value calculations.


## 2. Iteration log

### V1 — Naive churn model

I first built a churn prediction model using XGBoost. The purpose of this version was to reproduce the intuitive but flawed strategy: predict who is most likely to churn, then give the largest offer to the highest-risk users.

Holdout performance:

* ROC AUC: 0.63
* PR AUC: 0.5
* No-offer value: $4,276,892
* Churn baseline value: $4,233,947
* Churn baseline incremental value vs no offer: - $ 42,945
* Baseline Qini:

This strategy performed poorly because high churn risk did not necessarily imply high treatment responsiveness. In other words, many high-risk users were either lost causes or users for whom the offer did not create enough incremental value.

### V2 — Per-arm uplift model

Next, I moved from churn prediction to uplift modeling. I trained a T-learner: one separate churn model for each randomized offer arm. For each user, I predicted churn probability under arms 0, 1, 2, and 3.
I used T_learner because this approach is simple, transparent, and well suited as a first causal baseline after the naive churn model. 

For each paid arm, I defined uplift as:

`uplift_arm = predicted churn under no offer − predicted churn under offer arm`

A positive uplift means the offer is predicted to reduce churn. A negative uplift means the offer may increase churn, which is the “sleeping dog” case.

Per-arm Qini:

* Arm 1 Qini: 0.02
* Arm 2 Qini: 0.03
* Arm 3 Qini: 0.04
* Mixed weighted Qini: 0.03

### V3 — Expected net value per user-arm

I then converted uplift into expected net value:

`expected net value = uplift × annual_value − offer_cost`

This was important because a positive treatment effect is not automatically profitable. For example, a small uplift on a low-value subscriber may not justify a $15 concierge offer.

Since the budget was limited, I ranked candidate user-offer pairs by expected value per dollar:

`ROI score = expected net value / offer_cost`

I then greedily allocated offers until the $40,000 budget was exhausted. Users received no offer if all paid offers had negative or zero expected net value.

This changed the policy from “send the biggest offer to the riskiest users” to “spend each dollar where it creates the most incremental retained value.”

Holdout result before manual sleeping-dog rules:

* Uplift allocation value: $ 4,360,565
* Uplift allocation incremental value vs no offer: $ 83,674
* Uplift allocation incremental value vs churn baseline: $126,619
* Arm mix: arm 0 60%, arm 1 30.5%, arm 2 9%, arm 3 0.5%

### V4 — Sleeping-dog business rule

I considered a sleeping dog to be a customer for whom treatment is expected to be harmful or economically unattractive. The uplift model naturally avoids users whose predicted net value is negative. In addition, I tested a conservative EDA-derived business rule that excluded several segments showing negative empirical treatment effects

* `active_days_30d <= 1`
* `support_tickets_90d >= 4`
* `tenure_months > 21`

After applying this rule, the holdout result was:

* Value with manual sleeping-dog rule: $4,360,899.04
* Uplift allocation incremental value vs no offer: $84,007.36
* Uplift allocation incremental value vs churn baseline: $126,952.48


## 3. Sleeping dogs

I defined a sleeping dog as a user-offer pair where the predicted uplift was negative or economically unattractive. In practical terms, this means the model expected the offer either to increase churn or to produce too little value relative to cost.

The main protection against sleeping dogs was the expected net value filter. A user only received a paid offer if the best available offer had positive expected net value. Since:

`expected net value = uplift × annual_value − cost`

a negative uplift cannot produce positive expected net value. Therefore, the allocation naturally avoided user-offer pairs where the offer was predicted to be harmful.

In addition, I explored potential sleeping-dog segments manually. The EDA suggested that users with very low activity, many support tickets, or long tenure could be less responsive or negatively affected by outreach. I tested these as a conservative business rule and compared the holdout value before and after applying the rule.


## 4. Budget call: what if the budget were halved?

If the budget were reduced from $40,000 to $20,000, I would not proportionally reduce every offer. Instead, I would rerun the same allocation procedure with the lower budget and keep only the highest expected value-per-dollar candidates.

In practice, I would expect the halved-budget policy to shift toward:

* fewer concierge offers;
* more cheap nudges where uplift is positive;
* only the strongest discount/concierge cases;
* no offers for marginal positive-value users.

This is because under a tighter budget, the opportunity cost of each dollar increases. The decision rule should prioritize ROI first, then absolute value among similarly efficient candidates.


## 5. Production transfer: Databricks + AWS

### Feature pipeline

Raw event, subscription, billing, support, and product usage data would land in S3. Databricks jobs would transform these into a feature table using Delta Lake. The feature set would include only pre-treatment features available before offer assignment, such as tenure, recent activity, support history, price tier, autopay status, acquisition channel, prior offers, and MRR.

### Model training and scoring

The uplift models would be trained in Databricks using MLflow for experiment tracking, model versioning, and reproducibility. The scoring job would run on a regular cadence, for example weekly or daily depending on campaign frequency.
The allocation job would enforce the total budget constraint before writing the final campaign audience.

### Monitoring

I would monitor the following metrics:

   * feature missingness;
   * distribution drift;
   * changes in activity, MRR, support tickets, and acquisition mix.
   * uplift/Qini on future randomized holdouts;
   * calibration of predicted treatment effects;
   * stability of arm mix;
   * share of users predicted as sleeping dogs.
   * budget used;
   * retained revenue;
   * incremental retention;
   * ROI by offer arm;
   * cost per incremental retained subscriber.


I would retrain the model when:

* uplift/Qini decays materially;
* observed ROI by arm drops below expectation;
* the product or pricing changes;
* the customer mix changes significantly;
* campaign fatigue appears, especially among users with many prior offers.

I would also keep a small randomized exploration group to continue measuring treatment effects and prevent the model from becoming stale or overly exploitative.


### Finance

For Finance, I would focus on ROI and budget efficiency:

* total spend;
* incremental retained revenue;
* net incremental value;
* ROI by offer;
* sensitivity to budget size.

The key message would be:

“The model does not spend the full budget blindly. It spends only where expected incremental retained value exceeds offer cost, and prioritizes the highest return per dollar.”

### Product 

For Product, I would focus on which offer works for whom:

* which user segments respond to nudges;
* which users require discounts or concierge outreach;
* which users should not be contacted;
* whether cheaper offers can replace expensive outreach for some users.

The key message would be:

“The best offer is not the same for every user. Some users are persuaded by a low-cost nudge, some need stronger intervention, and some should receive no offer because outreach may be wasteful or harmful.”


## 6. What I would do if I had more time

There are several areas that I would explore further to improve both predictive performance and business value.

Improve the uplift estimation

I used a T-learner because it is simple, interpretable, and provides separate treatment-effect estimates for each offer. Given more time, I would compare it with other causal learning approaches such as X-learners, causal forests, or doubly robust learners. I would also perform hyperparameter tuning using cross-validation to improve treatment-effect estimation while avoiding overfitting.

Improve the budget optimization

The current allocation uses a greedy value-per-dollar heuristic after selecting the highest expected-value offer for each customer. While computationally efficient, it is not guaranteed to find the globally optimal allocation. Given more time, I would formulate the allocation as a constrained optimization problem (for example, a multiple-choice) to maximize expected net value under the budget constraint.

Better identification of sleeping dogs

The current sleeping-dog filter is based on exploratory data analysis and was intentionally designed as a conservative business rule. I would replace these manually selected thresholds with a model-driven approach by estimating treatment effects within individual offers and identifying customers whose predicted uplift is consistently negative. I would also quantify the uncertainty of these estimates to avoid excluding customers based on noisy observations.

Production readiness

For a production deployment, I would package the training and scoring pipeline into a reproducible workflow with automated feature generation, model versioning, scheduled retraining, monitoring of treatment effectiveness, and continuous tracking of business KPIs such as incremental retained revenue, budget utilization, and offer mix.