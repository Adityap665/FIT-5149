# FIT5149 Assessment 1

Collaborative development repository for the **Sanctioned Loan Amount Prediction** assignment.

> This repository contains course-provided data and templates for authorised group
> collaboration. Keep it private while the assessment is active and do not redistribute
> its contents. The assignment specification itself is intentionally excluded.

## Current status

The development notebook contains:

- data-quality and schema checks;
- train/test missingness and distribution-shift comparisons;
- target associations with missingness, numeric variables, and categorical variables;
- treatment of encoded missing values;
- investigation of the corrupted `property_age_years` field;
- feature engineering based on lending ratios and application dates;
- five-fold internal cross-validation;
- initial Ridge, Random Forest, and Histogram Gradient Boosting models;
- out-of-fold error diagnostics;
- a feature-engineering ablation using the same validation folds.

`OUR_PART_REPORT_NOTES.md` records the current evidence and decisions for the EDA,
preprocessing, feature-engineering, and validation sections of the report.

The final model and Kaggle submission have not yet been selected.

## Local setup

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the packages in `requirements.txt`.
4. Open and run `assignment_1_from_scratch.ipynb` from top to bottom.

## Collaboration workflow

- Create a short-lived branch for each change.
- Keep data preparation inside model pipelines to prevent leakage.
- Use the fixed validation folds when comparing models.
- Record every reported result in reproducible code.
- Review changes before merging into `main`.
- Do not commit credentials, the assignment specification, generated submissions, or final reports.

## Academic integrity

All contributors are responsible for following the unit's collaboration and generative-AI rules. Any permitted AI assistance must be declared in the submitted documentation.
