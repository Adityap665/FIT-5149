# Development archive

These files explain how we reached the selected model. They are kept for discussion,
report evidence, and rerunning comparisons. They are not extra final submissions.

- `FIT5149_A1_analysis.ipynb`: EDA, cleaning evidence, early model comparisons, and attempts 1/2.
- `FIT5149_A1_final_clean.ipynb`: Aditya's earlier CatBoost implementation for attempts 2/3.
- `FIT5149_A1_attempt5.ipynb`: separate three-seed candidate for the last experimental upload.
- `attempt5_experiments.py`: compares the last variations and exports/verifies attempt 5.
- `loan_experiments.py`: later model comparisons and optional new CSV export.
- `prepare_final_notebook.py`: helper to build a new standalone notebook. It is not needed
  to run the current final notebook and does not reproduce its later handwritten commentary.
- `verify_final_notebook.py`: fresh-kernel check of the final implementation.
- `build_report.py`: builds the editable Word report from checked data and saved results,
  without changing or training models. Requires python-docx, Pillow, pandas and NumPy.
- `report_assets/`: figure generated from the final model's verified held-out errors.
- `model_checks/*.json`: exact settings, validation scores, and the final verification record.
- `model_checks/cache/`: local saved validation predictions, ignored by Git.
- `.autosave_backup/`: local recovery copy of a notebook recreated by an open editor
  during the move. Its kernel metadata was kept in the organised analysis notebook.

The current final notebook is one folder above: `../FIT5149_A1_final.ipynb`.
The input data also stay one folder above. Only path and output-location changes were
made to the older notebooks during organisation; their models and saved outputs were kept.
Rerun CSVs have a `reproduced_` prefix in `../submissions/` to protect original attempts.

Do not include this whole folder in the final implementation ZIP.
