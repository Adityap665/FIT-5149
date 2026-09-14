# Report evidence and development history

These files explain how we reached the selected model. They are kept for discussion,
report evidence, and rerunning comparisons. They are not extra final submissions.

- `FIT5149_A1_analysis.ipynb`: EDA, cleaning evidence, early model comparisons, and attempts 1/2.
- `FIT5149_A1_final_clean.ipynb`: the main report-evidence notebook. It now recalculates
  the data tables, preprocessing comparisons, candidate comparisons, final diagnostics
  and Figure 1. It uses the selected attempt 4, not the old single CatBoost. The previous
  version is preserved in Git commit `fc3a1ae` and an ignored local backup.
- `FIT5149_A1_attempt5.ipynb`: separate three-seed candidate for the last experimental upload.
- `attempt5_experiments.py`: compares the last variations and exports/verifies attempt 5.
- `loan_experiments.py`: later model comparisons and optional new CSV export.
- `prepare_final_notebook.py`: helper to build a new standalone notebook. It is not needed
  to run the current final notebook and does not reproduce its later handwritten commentary.
- `verify_final_notebook.py`: fresh-kernel check of the final implementation.
- `build_report.py`: the earlier Word-layout builder. It now reads the tables and plot
  exported by `final_clean` and writes a separate rebuild draft. Do not treat it as the
  analysis source or use its five-page draft in place of the user-edited report and appendix.
- `report_assets/`: original report illustration, retained as a historical asset.
  The reproducible replacement is `../report/evidence/figure_1_error_by_request.png`.
- `model_checks/*.json`: exact settings, validation scores, and the final verification record.
- `model_checks/cache/`: local saved validation predictions, ignored by Git.
- `.autosave_backup/`: local recovery copy of a notebook recreated by an open editor
  during the move. Its kernel metadata was kept in the organised analysis notebook.

The current final notebook is one folder above: `../FIT5149_A1_final.ipynb`.
The input data also stay one folder above. The original analysis and attempt 5 notebooks
are historical records. `final_clean` has intentionally been updated into the report
evidence notebook. Its outputs go in `../report/evidence/` and do not overwrite the
original submissions or Word/PDF documents.

To verify the report, restart and run all cells in `final_clean`. The main comparisons
are fitted again. The optional broader-search switch is off by default and clearly
labels the archived records. Kaggle scores are externally confirmed observations.

Do not include this whole folder in the final implementation ZIP indiscriminately.
Staff recommends including the EDA and analysis as supporting material, but not every
rejected experiment. Keep the report notebook's section 13 helper and JSON records if
you include that notebook, and omit local caches and recovery copies.
