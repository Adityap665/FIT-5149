# Generated evidence for the report

Start with `../../development/FIT5149_A1_final_clean.ipynb`, not these individual files.
That notebook is the source. The CSVs and figure here are its generated outputs, so
they do not need separate manual maintenance. Do not edit a calculated score here.

The main model calculations were rerun on 14 September 2026. The run took about
26 minutes on Eric's machine. The report-audit cell was then revised and tested
separately against those newly generated tables to display outdated Word values.
Running the notebook from top to bottom performs both the calculations and the audit.

## Main report outputs

| Output | Notebook section |
|---|---|
| `table_1_missing_values.csv` | 3 |
| `table_2_preprocessing.csv` | 7 |
| `table_3_model_comparison.csv` | 12 |
| `table_4_partition_stability.csv` | 12 |
| `table_5_permutation.csv` | 9 |
| `table_6_kaggle_results.csv` | 14 |
| `figure_1_error_by_request.png` | 10 |
| `word_table_audit.csv` | 16 |
| `provenance.json` | 16 |

The remaining small tables show supporting calculations, fold scores and model
settings. Row-level OOF predictions, fold assignments and the recreated attempt 4
are local generated outputs and are ignored by Git. They can be regenerated.

## What the verification found

The graph and selected-model results match the earlier report. All 10,000 recreated
attempt 4 predictions match its original CSV byte for byte. Original submissions,
the root final notebook and the Word/PDF files were not changed by this run.

Four cells in the current Word tables need updating:

| Location | Old report value | Recalculated value |
|---|---:|---:|
| Preprocessing table, RF with property_ref | 35,973 | 35,994 |
| Preprocessing table, RF without property_ref | 35,965 | 35,956 |
| Model comparison, RF without property_ref | 35,965 | 35,956 |
| Model comparison, expected-tier HGB | 34,625 | 34,658 |

The corresponding prose also needs syncing: the RF difference is about AUD 38,
not AUD 8. The current hard-tier HGB has mean fold RMSE about AUD 39,459 and mean
fold MAE about AUD 14,425. The historical attempt 1 log and this RF rerun both
round to AUD 35,956. Older intermediate scores should not be described as this run.
These recalculated results also match the current saved analysis-notebook outputs.

`word_numeric_tables_match` is currently false in the provenance record. This is
deliberate: the notebook does not hide that the report needs a numeric update.
Rerun its last cell after updating Word. The table audit does not replace reviewing
the prose, figure placement or PDF layout.

## What can and cannot be recalculated

- The main data tables, validation scores, permutation checks and error plot are
  computed from the supplied data by code visible in the notebook.
- Kaggle public scores are observations confirmed by the team. Hidden test targets
  are unavailable, so those scores cannot be calculated locally.
- The broader 15-configuration search is shown as an archived log in section 13.
  Its data fingerprint and mean fold scores are checked, but it is not silently
  presented as a new run. Turn on the clearly labelled switch to retrain that search.
- The earlier report-layout builder reads these outputs. It does not own the analysis
  and does not overwrite the team's current Word report or appendix.
