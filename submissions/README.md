# Kaggle submission history

| File | Attempt | Confirmed public RMSE |
|---|---|---:|
| `submission_rf_attempt1.csv` | 1 | 35,514.14046 |
| `submission_catboost_attempt2.csv` | 2 | 33,510.57914 |
| `submission_catboost_twostage_attempt4.csv` | 4 | 33,096.96156 |
| `submission_catboost_seedavg_attempt5.csv` | 5 | 33,119.70974 |

No separate attempt 3 CSV was present when this folder was organised. Kaggle showed
the same public score as attempt 2; this alone does not prove the files were identical.

The root `../submission.csv` is byte-for-byte identical to attempt 4 and is the file
paired with the current final notebook.

Attempt 5 is an experimental equal average of three two-stage models. Validation is
effectively tied with attempt 4, not clearly better. Its standalone notebook is
`../development/FIT5149_A1_attempt5.ipynb`. Its confirmed public score was slightly worse
than attempt 4, which remains paired with the root final notebook and `submission.csv`.

Keep original submitted files unchanged. Older notebooks now save rerun predictions
with a `reproduced_` prefix here, ignored by Git. These are not new Kaggle attempts.
See `../report/KAGGLE_ATTEMPTS.md` for the full history and validation evidence.
