# FIT5149 Assessment 1

Collaborative development repository for the **Sanctioned Loan Amount Prediction** assignment.

> This repository contains course-provided data and templates for authorised group
> collaboration. Keep it private while the assessment is active and do not redistribute
> its contents. The assignment specification itself is intentionally excluded.

## Current status

The development notebook contains:

- data-quality and schema checks;
- train/test missingness and distribution-shift comparisons;
- annual and monthly income consistency checks;
- categorical comparisons using sanction and rejection rates;
- an identifier audit for `property_ref`;
- target associations with missingness, numeric variables, and categorical variables;
- treatment of encoded missing values;
- investigation of the corrupted `property_age_years` field;
- feature engineering based on lending ratios and application dates;
- five-fold internal cross-validation;
- initial Ridge, Random Forest, and Histogram Gradient Boosting models;
- out-of-fold error diagnostics;
- a feature-engineering ablation using the same validation folds;
- a `property_ref` feature-selection check;
- a small hyperparameter comparison for the two leading tree models;
- permutation evidence for the most influential variables;
- the first Random Forest Kaggle attempt, with a Public RMSE of 35,514.14046 and
  position 37 when submitted.

`Eric_Report_notes.md` records the current evidence and decisions for the EDA,
preprocessing, feature-engineering, and validation sections of the report.

The final model and Kaggle submission have not yet been selected.

## Local setup

1. Clone the repository and open a terminal in its folder.
2. Create and activate a Python 3.11 environment.
3. Install the packages with `python -m pip install -r requirements.txt`.
4. Start Jupyter from the activated environment.
5. Open `FIT5149_A1_analysis.ipynb`, restart the kernel, and run every cell from top to bottom.

The recorded notebook results were produced with the exact package versions in
`requirements.txt`.

## Collaboration workflow

- Create a short-lived branch for each change.
- Keep data preparation inside model pipelines to prevent leakage.
- Use the fixed validation folds when comparing models.
- Record every reported result in reproducible code.
- Name Kaggle files as `submission_<model>_attempt<number>.csv` and record the internal
  validation result before uploading them.
- Commit only the prediction files that were actually uploaded to Kaggle.
- Review changes before merging into `main`.
- Do not commit credentials, the assignment specification, unused prediction files, or final reports.

## Academic integrity

All contributors are responsible for following the unit's collaboration and generative-AI rules. Any permitted AI assistance must be declared in the submitted documentation.
