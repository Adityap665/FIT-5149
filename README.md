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
  development/              Report evidence, earlier analysis and experiment tools
    FIT5149_A1_analysis.ipynb
    FIT5149_A1_final_clean.ipynb  Main notebook for checking the report
    FIT5149_A1_attempt5.ipynb
    attempt5_experiments.py
    loan_experiments.py
    prepare_final_notebook.py
    verify_final_notebook.py
    model_checks/           Saved scores, checks, and local caches
  submissions/              Original Kaggle attempt CSVs
  report/                   Editable Word report, PDF, notes and attempt history
  forms/                    Original Word templates
```

There are still four working notebooks. No additional report notebook was added.
Use `development/FIT5149_A1_final_clean.ipynb` to reproduce the report's numerical
evidence and Figure 1. It now includes the earlier comparisons, selected attempt 4,
attempt 5 comparison, and a check that attempt 4 is reproduced exactly.
The root final notebook remains the standalone selected-model implementation.
The analysis notebook preserves the original EDA and development history.
The empty local `Untitled.ipynb` draft is also preserved under `development/` and ignored by Git.

The Word and matching PDF in `report/` contain the five-page report with attempt 4
results. The user-edited project plan and work-division appendix is included on page 6.
See `report/README.md` before treating the PDF as the complete submission.
The files in `forms/` are templates, not completed or signed forms.
The repository as a whole is not the final implementation ZIP.

## Last experimental attempt

`submissions/submission_catboost_seedavg_attempt5.csv` is a separate candidate that
averages three versions of the two-stage model, using seeds 42, 73, and 2026.
Its implementation is `development/FIT5149_A1_attempt5.ipynb`.
Validation is effectively tied with attempt 4: slightly worse on the original split
and slightly better on the second. Its confirmed public RMSE was 33,119.70974, slightly
worse than attempt 4's 33,096.96156. We retain the simpler attempt 4 implementation.
The root final notebook and `submission.csv` still reproduce attempt 4.
The attempt 5 notebook was executed from a fresh kernel and reproduced its CSV exactly.
See `report/KAGGLE_ATTEMPTS.md` for the full comparison.

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

`development/FIT5149_A1_final_clean.ipynb` has been updated, with the team's agreement,
from Aditya's attempts 2/3 implementation into the report-evidence notebook. The old
version is preserved in Git commit `fc3a1ae` and a local ignored recovery copy.
The selected final model is attempt 4, not the most recent upload, attempt 5.

## Reproduce the report evidence

Open `development/FIT5149_A1_final_clean.ipynb`, restart its kernel and run all cells.
It recalculates all six main report tables and Figure 1. Main model comparisons are
trained again; their scores are not copied from saved JSON files. The verification run
took about 26 minutes on Eric's computer, so allow time for the complete comparison.
The exported tables, row-level OOF predictions, plot and source record are placed in
`report/evidence/`. Its table of contents maps each report section to the calculation.

Kaggle public scores are clearly labelled as externally confirmed submission results:
they cannot be recalculated locally without the hidden targets. The broader 15-model
search is separately labelled as an archived log, with an optional full rerun switch.
It is not substituted for the freshly calculated main report tables.

This notebook does not replace the current Word/PDF, appendix or original CSVs.
Its regenerated attempt 4 CSV stays in `report/evidence/` and must match the original.
If Jupyter already has the old notebook open, close it without saving that stale tab
and reopen the updated file before running it.

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

- `development/FIT5149_A1_final_clean.ipynb`: one place to reproduce the report evidence.
- `development/FIT5149_A1_analysis.ipynb`: original EDA, earlier model comparisons, and historical attempts.
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
root. They find the input CSVs in the root. The analysis and attempt 5 notebooks write
rerun predictions with a `reproduced_` prefix in `submissions/`. The report-evidence
notebook writes only to `report/evidence/`. Original attempt CSVs are preserved.

The development requirements include XGBoost 3.2.0 because it was evaluated. The final
selected model uses CatBoost only. Keep rejected experiments separate from the small
final implementation. Staff's forum clarification also recommends including the EDA
and analysis notebook as supporting material in the code submission; rejected model
implementations are not all required. Their comparison still belongs in the report.

## Final handoff

The final ZIP should contain the final notebook, its matching `submission.csv`, this
README, requirements, the signed cover sheet, the completed AI declaration, and the
AI chat-history file. Include the report-evidence notebook and supporting EDA as
recommended by the staff clarification. Preserve the folders and the helper/results
used by section 13 if including the report-evidence notebook. Do not ZIP the entire
repository with Git files, caches and recovery copies.
Check that the selected Kaggle submission matches this model.

The report is submitted separately as `group43_ass1_report.pdf`. Include both members'
actual Kaggle usernames, TEAM ERIC-ADITYA, the comparison results, model limitations,
and the completed work-division statement. The implementation files do not replace
the report or the completed and signed forms.
