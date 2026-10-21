# Model card: loan default model

**Data:** 682 loans from the Czech bank (1993-1998), 76 defaults (11.1%). Default = status B (finished, not paid) or D (running, in debt).
**Features:** loan terms, account and owner attributes, district demographics, and the account's transactions in the 365 days BEFORE the loan. Nothing after the loan date.
**Chosen model:** logistic (rule fixed in advance: logistic unless boosting is significantly better).

## Cross-validated performance (5-fold x 10 repeats, 95% bootstrap CI)

| model | ROC AUC | PR AUC | Brier | defaults caught in riskiest 20% |
|---|---|---|---|---|
| rule_min_balance | 0.795 [0.742, 0.845] | 0.500 [0.392, 0.601] | n/a | 51% |
| logistic | 0.894 [0.854, 0.929] | 0.666 [0.565, 0.755] | 0.060 | 74% |
| boosting | 0.905 [0.864, 0.942] | 0.707 [0.612, 0.791] | 0.055 | 79% |

PR AUC of a random model = default rate = 0.111.

Paired comparisons (difference in ROC AUC, 95% CI, p-value):

- logistic_vs_rule: +0.10 [+0.05, +0.15], p=0.000
- boosting_vs_logistic: +0.01 [-0.01, +0.03], p=0.304

## Time-based check

Trained on 328 loans granted up to 1996, tested on 196 loans from 1997 (26 defaults): logistic ROC AUC 0.957 [0.912, 0.987], boosting 0.946 [0.908, 0.975].
1998 loans are excluded from testing: they are too recent to have defaulted (label not mature).

Why this can beat cross-validation: the 1997 defaults are mostly status D (23 D vs 3 B), i.e. clients already in debt early in the loan, whose weak pre-loan history makes them easier to spot. Over all years it's 45 D vs 31 B. So the higher score reflects an easier test mix, not a model that improves over time; with this few defaults the interval is also wide.

## Leakage check

Same model on features that include transactions AFTER the loan: ROC AUC 0.997 [0.994, 0.999]. The jump is the outcome leaking into the features, which is why the window stops at the loan date.

## Calibration (out-of-fold, 5 equal-size bins)

| model | predicted | observed | n |
|---|---|---|---|
| logistic | 0.00 | 0.00 | 137 |
| logistic | 0.01 | 0.01 | 137 |
| logistic | 0.04 | 0.05 | 136 |
| logistic | 0.10 | 0.09 | 136 |
| logistic | 0.40 | 0.41 | 136 |
| boosting | 0.00 | 0.01 | 137 |
| boosting | 0.01 | 0.01 | 137 |
| boosting | 0.03 | 0.03 | 136 |
| boosting | 0.07 | 0.07 | 136 |
| boosting | 0.38 | 0.44 | 136 |

## Limitations

- 76 defaults is a small sample: intervals are wide, and small differences between models are noise.
- Running loans (status C) may still default, so some 'good' labels are wrong, more so for recent loans.
- One bank, 1990s Czech Republic: not representative of other populations or today.
- Not for real credit decisions; gender is in the features, which real lending law may prohibit.
