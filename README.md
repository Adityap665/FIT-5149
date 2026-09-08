# FIT5149 Assessment 1

Collaborative development repository for the **Sanctioned Loan Amount Prediction** assignment.

> Keep this repository private while the assessment is active. Course-provided data,
> assessment documents, AI declarations, reports, and Kaggle submissions are intentionally
> excluded from version control.

## Current status

The initial development notebook contains:

- data-quality and schema checks;
- treatment of encoded missing values;
- investigation of the corrupted `property_age_years` field;
- feature engineering based on lending ratios and application dates;
- five-fold internal cross-validation;
- initial Ridge, Random Forest, and Histogram Gradient Boosting models;
- out-of-fold error diagnostics.

The final model and Kaggle submission have not yet been selected.

## Local setup

1. Clone the repository.
2. Create and activate a Python environment.
3. Install the packages in `requirements.txt`.
4. Place `training_set.csv` and `kaggle_test_X.csv` in the repository root. These files are ignored by Git and must be obtained through the unit's authorised channel.
5. Open and run `assignment_1_from_scratch.ipynb` from top to bottom.

## Collaboration workflow

- Create a short-lived branch for each change.
- Keep data preparation inside model pipelines to prevent leakage.
- Use the fixed validation folds when comparing models.
- Record every reported result in reproducible code.
- Review changes before merging into `main`.
- Do not commit credentials, raw data, assignment documents, generated submissions, or final reports.

## Academic integrity

All contributors are responsible for following the unit's collaboration and generative-AI rules. Any permitted AI assistance must be declared in the submitted documentation.

