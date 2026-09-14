# Kaggle attempts and final model checks

## Attempt record

| Attempt | Model | Public RMSE | Status |
|---|---|---:|---|
| 1 | Random Forest | 35,514.14046 | Submitted earlier |
| 2 | Original CatBoost rate model | 33,510.57914 | Submitted by the team |
| 3 | Original CatBoost, clean implementation | 33,510.57914 | Matched the earlier best score |
| 4 | Two-stage CatBoost | 33,096.96156 | Submitted by Eric; position 19 in his screenshot |
| 5 | Equal average of three two-stage CatBoost models | 33,119.70974 | Submitted by Eric; attempt 4 remained the best public score |

The submitted file is `submissions/submission_catboost_twostage_attempt4.csv`. The current
`FIT5149_A1_final.ipynb` must reproduce exactly the same predictions as `submission.csv`.
Attempt 4 improved the public RMSE by 413.61758 compared with the previous best.
Position 19 is the position at submission, not a final rank. Marks use the private score.
Confirm the next unused attempt number with Aditya if he uploads anything else.

## Why we tried two stages

About 26.8% of training applications have a sanctioned amount of zero. This suggests
two related questions: whether an application is approved, and how much is approved
when it is successful. We trained a CatBoost classifier for the first question and a
CatBoost regressor on the positive cases for the second.

For each application, we multiply its approval probability by the expected positive
sanction rate and the amount requested. We do not force every prediction into an
approve/reject category or into one of the observed tiers. Keeping probabilities lets
the prediction reflect uncertainty.

Both models are fitted inside each validation training fold. The positive-case
regressor never sees validation labels. Cleaning and predictor values match the earlier
CatBoost implementation. The final model uses all 49 engineered inputs, including
12 categorical inputs.

## Candidate comparison

All results below use the original five stratified folds with split seed 42.
The training seed is 42. Errors are measured in AUD. We compare global out-of-fold
RMSE and also report mean fold RMSE for consistency with earlier notebook tables.
The same candidate ranks first under both summaries.

| Candidate | Mean fold RMSE | Global OOF RMSE | MAE |
|---|---:|---:|---:|
| Two-stage CatBoost (selected) | 33,958.77 | 33,975.51 | 16,850.52 |
| Two-stage CatBoost, depth 7 | 34,015.72 | 34,032.33 | 16,479.20 |
| Two-stage CatBoost, longer training | 34,073.81 | 34,093.19 | 16,451.55 |
| Original CatBoost rate model | 34,099.76 | 34,117.19 | 17,267.27 |
| Two-stage CatBoost, 12 predictors | 34,120.28 | 34,137.86 | 16,960.55 |
| Two-stage CatBoost, depth 3 | 34,131.30 | 34,145.86 | 17,164.10 |
| CatBoost rate, longer training | 34,201.92 | 34,219.29 | 17,311.31 |
| CatBoost rate, shallower trees | 34,204.55 | 34,222.35 | 17,582.27 |
| Two-stage XGBoost | 34,335.35 | 34,349.99 | 16,739.45 |
| CatBoost tier probabilities | 34,503.69 | 34,520.02 | 17,922.42 |
| CatBoost rate, request-squared weights | 34,606.01 | 34,620.80 | 17,877.45 |
| CatBoost rate, four core predictors | 34,680.71 | 34,696.72 | 17,609.38 |
| Two-stage CatBoost, four core predictors | 34,689.04 | 34,705.12 | 17,122.57 |
| XGBoost rate | 34,779.85 | 34,796.54 | 17,598.16 |
| CatBoost predicting AUD directly | 35,348.16 | 35,363.15 | 18,431.14 |

These are development and selection results, not estimates from a separate test set.
The four-input and 12-input feature sets were informed by the earlier EDA and Random
Forest evidence, so their assessment is exploratory. Search and feature selection can
make the selected score optimistic.

Two fixed 50/50 combinations were checked using the saved validation predictions:
the selected two-stage model with the original CatBoost (global RMSE 33,995.84), and
with two-stage XGBoost (34,069.61). Neither improved on the selected model's 33,975.51.
We did not tune a large grid of blend weights.

## Check with another partition

| Model | Mean fold RMSE, split 42 | Mean fold RMSE, split 2026 |
|---|---:|---:|
| Original CatBoost | 34,099.76 | 34,323.40 |
| Two-stage CatBoost | 33,958.77 | 34,119.05 |
| Reduction | 140.98 | 204.35 |

The selected model improves each paired fold in both partitions. This is a useful
stability check, but these runs reuse the same applications and are not independent
experiments. It does not establish statistical significance or guarantee a public or
private Kaggle improvement. We have not reached RMSE 33,000 in internal validation.

We prepared one new submission because this candidate had the clearest support.
Another upload with the same predictions would not test a different model.

## Reproduce the comparison

All paths below are relative to the repository root. Install `requirements.txt`, then run from that root:

```bash
python development/loan_experiments.py
python development/loan_experiments.py --models catboost_current catboost_two_stage --split-seed 2026
```

Each completed model stores its exact settings, fold scores, feature list, data SHA256,
and runtime in `development/model_checks/`. Local OOF arrays are cached in `development/model_checks/cache/`.
If caches are absent, the calculations run again.

The standalone final notebook includes the selected model's own error breakdown and
three-repeat permutation checks for five prespecified inputs. It does not depend on
the experiment runner. Its fresh-kernel verification is recorded in
`development/model_checks/final_verification.json`.

## Final-model findings for the report

The selected model's global out-of-fold RMSE is AUD 33,975.51 and its MAE is
AUD 16,850.52. The RMSE is much larger because some applications have large errors.

The model still struggles with rejected applications. Their RMSE is about AUD 58,765,
compared with AUD 17,690 for approved applications. It sometimes predicts a positive
expected amount for a case that was actually rejected. This is a limitation when the
available information leaves the decision uncertain.

Credit-rating availability also matters. RMSE is about AUD 40,738 when the rating is
missing, compared with AUD 32,006 when it is available. The smallest requested-amount
group has RMSE of about AUD 9,014, while the largest has RMSE of AUD 59,394. Better
credit information and closer review of large or uncertain cases would be useful
before considering a real lending application.

The final model's permutation checks support the importance of credit rating and
co-applicant status. Shuffling them increases RMSE by approximately AUD 20,198 and
AUD 17,953 respectively. Income consistency has a smaller increase of about AUD 1,356.
These checks cover five prespecified inputs, not all features, and do not show causation.
Requested amount also determines the conversion to dollars, so shuffling it changes
the whole prediction process; its effect is not directly comparable to the earlier
Random Forest check, which kept the original requested-amount cap.

These numbers come from the selected model's own validation predictions, not from
the earlier Random Forest results. Attempt 4 has a confirmed public RMSE of AUD 33,096.96156.

## Attempt 5: one last experimental alternative

We tested five variations of attempt 4, plus an exact repeat of attempt 4 as a check.
The repeat matched the earlier validation result. None of the new variations improved
the original split's mean RMSE.

| Variation | Mean fold RMSE, split 42 |
|---|---:|
| Attempt 4, repeated | 33,958.77 |
| Equal average of seeds 42, 73, and 2026 | 33,989.17 |
| Approval classifier with 600 iterations | 33,993.87 |
| Training weights proportional to requested amount | 34,089.24 |
| Squared-error approval model, depth 5 | 34,090.07 |
| Squared-error approval model, depth 3 | 34,411.71 |

The seed average was the closest alternative, so we checked it on the second partition
already used for attempt 4. The same rows are reused, so this is a stability check, not
an independent test or proof of significance.

| Model | Mean fold RMSE, split 42 | Mean fold RMSE, split 2026 |
|---|---:|---:|
| Attempt 4 | 33,958.77 | 34,119.05 |
| Three-seed average | 33,989.17 | 34,081.52 |

The new candidate is worse by AUD 30.40 on the first partition and better by AUD 37.54
on the second. Averaging these two mean scores gives only about AUD 3.57 improvement.
This is effectively a tie, not evidence of a clear gain. Its global OOF RMSE is
34,005.33 on split 42 and 34,092.48 on split 2026. We do not claim the candidate is
better or that it will cross the 33,000 threshold on Kaggle.

Eric requested one last available upload. We prepared the seed average as a separate
experimental alternative, while retaining attempt 4 as the current final implementation.
At preparation time, the new public score was still unknown; the confirmed result is recorded below.

For each seed, the approval model uses depth 5 and 1,200 iterations. The positive-rate
model uses depth 4 and 800 iterations. Both retain learning rate 0.03 and L2 regularisation
10. We average the three complete dollar predictions with equal weights, then round
to two decimals. We do not average the two stages separately or fit blend weights.
Feature engineering, data, and validation folds are unchanged.

Files, with paths relative to the repository root:

- `submissions/submission_catboost_seedavg_attempt5.csv`: candidate for the next upload.
- `development/FIT5149_A1_attempt5.ipynb`: standalone implementation of this candidate.
- `development/attempt5_experiments.py`: comparison, export, and verification helper.
- `development/model_checks/attempt5_*.json`: settings and measured results.

The attempt 5 notebook passed a fresh-kernel execution with all 11 code cells and no
errors. It reproduced the exported CSV byte for byte. The file has 10,000 checked rows
with the required identifiers, columns, and prediction bounds. This check is saved in
`development/model_checks/attempt5_verification.json`; it is a reproducibility check,
not evidence that the public score will improve.

To reproduce the comparison:

```bash
python development/attempt5_experiments.py
python development/attempt5_experiments.py --models seed_average --split-seed 2026
```

To rerun the standalone notebook verification on Eric's registered `fit5196` kernel:

```bash
python development/attempt5_experiments.py --verify-notebook submission_catboost_seedavg_attempt5.csv
```

Attempt 5 was subsequently submitted and scored 33,119.70974, compared with attempt 4's
33,096.96156. This is 22.74818 worse on the public leaderboard. The team stayed at
position 19 in Eric's screenshot. We retain the simpler attempt 4 implementation: the
seed average did not give a clear validation improvement, and its public score was
also slightly worse. The private results remain unknown.

The root `FIT5149_A1_final.ipynb` and `submission.csv` still belong to attempt 4.
The attempt 5 notebook and CSV remain in the development and submission-history folders.

## What Aditya can finish

- Use the final-model diagnostics and this comparison to write the modelling and
  limitations sections in normal language.
- Check the actual Kaggle usernames of both members and include them with
  TEAM ERIC-ADITYA and Group 43 in the report.
- Complete the cover sheet, AI declaration, AI chat history, and work-division statement.
- Confirm which Kaggle entries are nominated for the private evaluation.
- Assemble the implementation ZIP after confirming the matching Kaggle entry is nominated.
  Include the matching final model, not the older clean reference notebook.

Kaggle allows three submissions per day per team. Check remaining daily slots before
uploading. The local specification gives the deadline as 14 September 2026 at 23:55.
The PDF report is separate from the implementation ZIP.

## Method references

- CatBoost regression objectives: https://catboost.ai/docs/en/concepts/loss-functions-regression
- XGBoost categorical features: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- Model-selection bias and validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
