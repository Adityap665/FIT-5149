# FIT5149 Assessment 1 — Final Implementation

## Group
**Group ID:** 43

## Final model
The submitted implementation reproduces the team's final Kaggle model:

**CatBoost target-aware sanction-rate regression**

The model predicts the sanction rate for each application and converts that prediction back to a sanctioned dollar amount using `requested_amount_aud`.

## Files required to run
Place the following files in the same directory:

- `FIT5149_A1_final.ipynb`
- `training_set.csv`
- `kaggle_test_X.csv`
- `sample_submission.csv`

## Environment
Recommended environment:

- Python 3.11.9
- numpy
- pandas
- scikit-learn
- matplotlib
- catboost 1.2.10
- jupyter / jupyterlab

Install the required packages using:

```bash
pip install -r requirements.txt
```

## How to run
1. Open `FIT5149_A1_final.ipynb` in Jupyter Notebook or JupyterLab.
2. Make sure the three supplied CSV files are in the same directory as the notebook.
3. Restart the kernel.
4. Run all cells from top to bottom.
5. The notebook will:
   - validate the input files,
   - perform the required cleaning and feature engineering,
   - reproduce the five-fold validation of the selected CatBoost model,
   - retrain the final model using all labelled training data,
   - generate the final Kaggle prediction file.

## Output
Running the notebook creates:

`submission.csv`

This file contains:

- `application_id`
- `sanctioned_amount_aud`

The output is checked for row count, identifier order, missing predictions, finite values, non-negative predictions, and predictions not exceeding the requested loan amount.

## Reproducibility
The implementation uses:

- `RANDOM_STATE = 42`
- `N_SPLITS = 5`
- `StratifiedKFold(shuffle=True, random_state=42)`
- CatBoost with a fixed random seed

The notebook also prints package-version information at the end.

## Notes
Only the implementation of the final selected model is included, as required by the assignment instructions. Code for candidate models that were evaluated but rejected has been removed. Their validation results and comparison are presented in the written report.

## Final submission file
The notebook must reproduce the submitted Kaggle file:

`submission.csv`
