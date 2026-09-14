# FIT5149 Assessment 1: Group 43

Kaggle team: **TEAM ERIC-ADITYA**.

## Where to find everything

```text
FIT-5149/
  FIT5149_A1_final.ipynb     Current final implementation
  submission.csv           Predictions matching attempt 4
  README.md
  requirements.txt
  training_set.csv          Input data, kept here for the final notebook
  kaggle_test_X.csv
  sample_submission.csv
  development/              Earlier notebooks and experiment tools
    FIT5149_A1_analysis.ipynb
    FIT5149_A1_final_clean.ipynb
    loan_experiments.py
    prepare_final_notebook.py
    verify_final_notebook.py
    model_checks/           Saved scores, checks, and local caches
  submissions/              Original Kaggle attempt CSVs
  report/                   Report notes and attempt history
  forms/                    Original Word templates
```

There are three working notebooks. Use the final notebook in the root for the current
submission. The analysis notebook contains our EDA and development work. The older
final-clean notebook is Aditya's single-model implementation for attempts 2/3.
The empty local `Untitled.ipynb` draft is also preserved under `development/` and ignored by Git.

The notes in `report/` are drafting material, not the completed five-page PDF report.
The files in `forms/` are templates, not completed or signed forms.
The repository as a whole is not the final implementation ZIP.

## Current model and submission

The selected candidate is a two-stage CatBoost model:

1. Estimate the probability that an application receives a positive amount.
2. Estimate the sanction rate among approved applications.
3. Multiply the approval probability, expected positive rate, and requested amount.

We keep probabilities rather than assigning every application a hard approve/reject
decision. Predictions are limited to zero through the requested amount.

`FIT5149_A1_final.ipynb` is the standalone implementation for this model. It creates
`submission.csv`, which must match `submissions/submission_catboost_twostage_attempt4.csv`.
Attempt 4 was submitted: public RMSE 33,096.96156, position 19 in Eric's screenshot.
The position can change, and the private leaderboard determines the competition marks.
Nominate its Kaggle entry before using this implementation for the final assessment.

The earlier `development/FIT5149_A1_final_clean.ipynb` is retained as Aditya's attempt 2/3 reference.
It uses the original single-regressor CatBoost. When rerun, it now saves
`submissions/reproduced_catboost_clean.csv` so it cannot replace the final submission.
Use `FIT5149_A1_final.ipynb` to reproduce the new candidate.

## Validation evidence

Both candidates use the same five folds within each comparison. The training seed is
42 throughout. Changing the split seed checks sensitivity to the partition; it does
not create an independent test set or remove model-selection bias.

| Model | Mean fold RMSE, split 42 | Global OOF RMSE, split 42 | Mean fold RMSE, split 2026 |
|---|---:|---:|---:|
| Original CatBoost rate model | 34,099.76 | 34,117.19 | 34,323.40 |
| Selected two-stage CatBoost | 33,958.77 | 33,975.51 | 34,119.05 |

The improvement is modest. It does not establish a statistically significant gain
or guarantee a better Kaggle score. We selected the candidate from internal validation,
not from a new public leaderboard result. `report/KAGGLE_ATTEMPTS.md` records the comparison.

## Setup and execution

Use Python 3.11 and install the recorded dependencies:

```bash
python -m pip install -r requirements.txt
```

Place these files in the same folder:

- `FIT5149_A1_final.ipynb`
- `training_set.csv`
- `kaggle_test_X.csv`
- `sample_submission.csv`

Open the final notebook in Jupyter, select the Python environment containing CatBoost,
restart the kernel, and run all cells in order. On Eric's computer that kernel is
`Python (FIT5196)`. A teammate can select their own Python 3.11 kernel.

The notebook validates the input files, applies the same cleaning and features as
Aditya's earlier implementation, evaluates the chosen model, examines its errors and
five prespecified inputs, and then retrains on all labelled data. Checks run before
the prediction file is saved.

## Reproducibility

- Training random seed: 42.
- Main validation: five stratified folds, shuffle enabled, split seed 42.
- Approval model: CatBoost, Logloss, depth 5, 1,200 iterations, learning rate 0.03,
  L2 regularisation 10.
- Positive-rate model: CatBoost, RMSE, depth 4, 800 iterations, learning rate 0.03,
  L2 regularisation 10.
- CPU threads: 6.
- Verification environment: Python 3.11.15, NumPy 2.4.6, pandas 3.0.5,
  scikit-learn 1.9.0, CatBoost 1.2.10.

`development/model_checks/final_verification.json` records the fresh-kernel check and the SHA256
of the exact CSV. The final notebook has no dependency on the development scripts.

## Development files

- `development/FIT5149_A1_analysis.ipynb`: EDA, earlier model comparisons, and historical attempts.
- `report/Eric_Report_notes.md`: evidence for the EDA and preprocessing sections.
- `development/loan_experiments.py`: reproducible comparison of the later candidates.
- `development/model_checks/`: saved configurations, scores, and final-model diagnostics.
- `development/prepare_final_notebook.py`: creates the standalone selected-model notebook.
- `development/verify_final_notebook.py`: runs that notebook in a fresh kernel and compares its CSV.

To reproduce the small model comparison, run:

```bash
python development/loan_experiments.py
```

Run these commands from the repository root. The experiment runner saves scores in
`development/model_checks/` and new attempt CSVs in `submissions/`.

To verify the final notebook against the submitted attempt:

```bash
python development/verify_final_notebook.py FIT5149_A1_final.ipynb submissions/submission_catboost_twostage_attempt4.csv
```

The verification helper currently uses Eric's registered `fit5196` kernel. Aditya
can run the final notebook directly with his own environment.

Open development notebooks from `development/`, or start their kernel in the repository
root. They find the input CSVs in the root. Their rerun outputs use a `reproduced_`
prefix in `submissions/`; the original attempt CSVs are preserved. Existing notebook
outputs document the earlier runs, before this folder reorganisation.

The development requirements include XGBoost 3.2.0 because it was evaluated. The final
selected model uses CatBoost only. Rejected candidate code belongs in the development
repository, not in the final implementation ZIP.

## Final handoff

The final ZIP should contain the final notebook, its matching `submission.csv`, this
README, requirements, the signed cover sheet, the completed AI declaration, and the
AI chat-history file. Check that the selected Kaggle submission matches this model.

The report is submitted separately as `group43_ass1_report.pdf`. Include both members'
actual Kaggle usernames, TEAM ERIC-ADITYA, the comparison results, model limitations,
and the completed work-division statement. The implementation files do not replace
the report or the completed and signed forms.
