# EDA, preprocessing, feature engineering, and validation notes

These notes summarise the current reproducible results in
`FIT5149_A1_analysis.ipynb`. They are development notes rather than final report
copy. Every reported value should remain tied to notebook output after subsequent model
changes.

## Data integrity and coverage

- The labelled data contain 19,322 rows and 29 columns. The Kaggle test data contain
  10,000 rows and 28 columns.
- There are no duplicated rows, duplicated application IDs, or IDs shared between train
  and test.
- Both datasets cover 1 January 2023 to 30 December 2024. The date periods overlap, so a
  temporal holdout is not required by the observed collection ranges.
- Numeric train/test standardised mean differences are all at most 0.022 in absolute
  value. The largest categorical total-variation distance is 0.061 for `branch_id`.
  This indicates limited marginal distribution shift, although it does not rule out
  conditional shift.

## Missingness

The largest raw training missingness rates are:

| Feature | Train missing | Test missing |
|---|---:|---:|
| `property_age_years` | 32.40% | 32.62% |
| `monthly_income_aud` | 25.69% | 24.84% |
| `annual_income_aud` | 25.09% | 24.85% |
| `employment_sector` | 24.36% | 24.19% |
| `credit_rating` | 20.46% | 21.33% |

The train/test rates are close, with a maximum difference of 0.87 percentage points
among the listed fields. Missingness is nevertheless informative within the labelled
data. For example, mean sanctioned amount is AUD 17,627 higher when
`income_consistency` is absent and AUD 17,588 higher when
`existing_repayments_aud` is absent. These are descriptive associations and may reflect
other variables such as requested loan size.

Missingness is also systematic across applicant groups. `employment_sector` is absent
for 100% of retirees but 15.8% of salaried applicants, consistent with structural rather
than random missingness. Credit-rating missingness ranges from 11.1% for public-sector
applicants to 46.5% for retirees. The model therefore retains an aggregate missing-value
count and specific missingness indicators instead of relying only on imputed values.

## Invalid and corrupted values

- The value `-999` occurs 406 times in the labelled data: 100 repayments, 107
  co-applicant indicators, and 199 property values. These values are impossible for the
  documented variables and are converted to missing before fold-specific imputation.
- `property_age_years` is corrupted. All 11,524 rows in which it overlaps with annual
  income match `annual_income_aud` exactly, and it has no values between zero and 200.
  It is not interpreted as property age.
- An ablation found no improvement from using the corrupted field as an income backup:
  the Histogram Gradient Boosting RMSE was AUD 36,074 with the backup and AUD 36,065
  without it. Its numeric values are therefore dropped. Its missingness indicator is
  retained and evaluated separately because availability may still carry information.
- After sentinel cleaning, the checked monetary fields contain no nonpositive property
  values, negative repayments, nonpositive income, negative sanctioned amounts, or
  sanctions above the requested amount. Applicant ages are between 18 and 65.

## Income consistency

- Annual and monthly income are available together for 13,712 rows, or 70.97% of the
  labelled data. Annual income alone is available for 762 rows, monthly income alone for
  647 rows, and neither is available for 4,201 rows.
- When both are present, their Pearson correlation is approximately 1.000 and the median
  annual-to-monthly ratio is 11.999. The median relative difference between annual income
  and 12 times monthly income is 2.40%, and 84.52% of these rows are within 5%.
- Since the two fields mostly contain the same information, the current preprocessing
  creates one reconciled annual-income feature. The raw income fields can be checked in a
  simple comparison later if needed.

## Target structure and univariate EDA

- There are 5,177 zero-sanction applications, representing 26.79% of training rows.
- The rounded sanctioned/requested ratios take only eight values: 0, 0.65, 0.70, 0.75,
  0.80, 0.85, 0.90, and 1.00. This pattern is useful when interpreting the target and
  model errors.
- The strongest Pearson relationships with the target are requested amount (0.740),
  property value (0.712), existing repayments (0.563), credit rating (0.375), and
  co-applicant status (0.269).
- The corresponding Spearman relationships for requested amount, property value,
  repayments, credit rating, and co-applicant status are 0.601, 0.582, 0.481, 0.404,
  and 0.350.
- Categorical target differences are screening results only. `branch_id` has the largest
  spread in mean sanctioned amount across levels with at least 30 observations, but this
  may reflect different applicant and requested-amount mixes by branch.

## Categorical findings

- Credit rating has the clearest relationship with the outcome. The lowest credit band
  has a 70.9% rejection rate and a 19.6% mean sanction rate, compared with 12.3% and
  66.4% for the highest band.
- Salaried applicants have a 29.3% rejection rate, compared with 14.4% for retirees.
  These are raw group comparisons and do not control for income, age, or loan size.
- Regional applicants have a 30.4% rejection rate and a 49.4% mean sanction rate. Inner
  metropolitan applicants have a 22.0% rejection rate and a 56.9% mean sanction rate.
- Applicants with variable income have a 29.0% rejection rate, compared with 15.4% for
  steady income and 10.7% when income consistency is missing.
- Rejection rates by branch range from 11.1% to 40.0%. This is descriptive only because
  each branch can have a different mix of applicants and requested loan amounts.

## Preprocessing and feature engineering

- `application_id` is excluded from predictors and retained only for the final output.
- Dates are parsed explicitly as `DD/MM/YYYY` and represented by year plus cyclical month
  and day-of-year terms.
- Numeric values are median-imputed inside each training fold. Ridge additionally scales
  numeric inputs.
- Categories are imputed and encoded inside each fold. Unknown and rare levels are
  handled without fitting encoders on the Kaggle test data.
- `property_ref` is treated as categorical rather than continuous.
- Lending features include requested/property value, requested/reconciled annual income,
  repayments/reconciled monthly income, and property value minus requested amount.
- With Histogram Gradient Boosting held fixed, ratios plus missingness indicators improve
  mean RMSE from AUD 36,111 to AUD 36,065. The AUD 46 change is small and should be
  described as marginal rather than conclusive.

## Validation and current benchmark

Five shuffled stratified folds are used with random seed 42. Stratification uses the
eight observed sanction tiers, preserving the 26.8% zero-sanction group and the rare
high-sanction tiers in every fold. Every preprocessing step is fitted inside its training
fold, and all model families use identical precomputed folds.

| Model | Mean RMSE | RMSE SD | Mean MAE |
|---|---:|---:|---:|
| Random Forest | AUD 35,994 | AUD 1,447 | AUD 17,989 |
| Histogram Gradient Boosting | AUD 36,065 | AUD 1,283 | AUD 19,047 |
| Ridge | AUD 43,306 | AUD 887 | AUD 27,868 |

Random Forest currently leads Histogram Gradient Boosting by only AUD 71 mean RMSE. It
wins four of five paired folds, while Gradient Boosting wins one fold by AUD 693. The
difference is too small to declare either model final before a small tuning comparison.

RMSE remains the selection metric because it is the Kaggle metric and penalises costly
large-dollar errors. MAE is reported as an operationally interpretable alternative.
MAPE is unsuitable because 26.79% of targets are zero.

Current error analysis shows a major scale limitation: Random Forest RMSE rises from AUD
9,375 in the smallest requested-amount quintile to AUD 63,686 in the largest quintile.
The zero-sanction group also has RMSE of AUD 60,841. The next modelling stage should
check whether small tuning changes improve errors for zero sanctions and large loans.

## Handoff to the modelling work

The remaining work can stay close to the methods already used in class:

1. apply a small, clearly explained tuning search to Random Forest and Histogram Gradient
   Boosting using the existing folds;
2. compare the leading model with and without `property_ref`;
3. present simple feature-influence evidence using correlations, category rates, and an
   interpretable linear model already covered in class;
4. select the final model, train it on all labelled data, and create the submission file.

More complex two-stage or tier-classification models are not necessary unless the simple
changes above fail and there is a clear reason to add them.
