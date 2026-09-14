# Kaggle attempts and final model checks

## Attempt record

| Attempt | Model | Public RMSE | Status |
|---|---|---:|---|
| 1 | Random Forest | 35,514.14046 | Submitted earlier |
| 2 | Original CatBoost rate model | 33,510.57914 | Submitted by the team |
| 3 | Original CatBoost, clean implementation | 33,510.57914 | Matched the earlier best score |
| 4 | Two-stage CatBoost | Pending | CSV prepared locally, not uploaded by this workflow |

The prepared file is `submission_catboost_twostage_attempt4.csv`. The current
`FIT5149_A1_final.ipynb` must reproduce exactly the same predictions as `submission.csv`.
A new public score should only be entered here after the team actually submits the file.
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

Install `requirements.txt`, then run:

```bash
python loan_experiments.py
python loan_experiments.py --models catboost_current catboost_two_stage --split-seed 2026
```

Each completed model stores its exact settings, fold scores, feature list, data SHA256,
and runtime in `model_checks/`. Local OOF arrays are cached in `model_checks/cache/`.
If caches are absent, the calculations run again.

The standalone final notebook includes the selected model's own error breakdown and
three-repeat permutation checks for five prespecified inputs. It does not depend on
the experiment runner. Its fresh-kernel verification is recorded in
`model_checks/final_verification.json`.

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
the earlier Random Forest results. No public score is available for attempt 4 yet.

## What Aditya can finish

- Use the final-model diagnostics and this comparison to write the modelling and
  limitations sections in normal language.
- Check the actual Kaggle usernames of both members and include them with
  TEAM ERIC-ADITYA and Group 43 in the report.
- Complete the cover sheet, AI declaration, AI chat history, and work-division statement.
- Confirm which Kaggle entries are nominated for the private evaluation.
- Assemble the implementation ZIP after the new prediction file is uploaded and nominated.
  Include the matching final model, not the older clean reference notebook.

Kaggle allows three submissions per day per team. Check remaining daily slots before
uploading. The local specification gives the deadline as 14 September 2026 at 23:55.
The PDF report is separate from the implementation ZIP.

## Method references

- CatBoost regression objectives: https://catboost.ai/docs/en/concepts/loss-functions-regression
- XGBoost categorical features: https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html
- Model-selection bias and validation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
