# Report and working material

- `group43_ass1_report.docx`: editable report with the selected attempt 4 model.
- `group43_ass1_report.pdf`: matching, visually checked PDF with five assessed pages
  followed by the user-edited project-plan appendix on page 6.
- `group43_project_plan_work_division.docx`: one-page editable appendix with the agreed
  50/50 contribution, task descriptions and project progress. Its current user-edited
  content is included on page 6 of the report. Future edits must be synced to the report.

- `Eric_Report_notes.md`: evidence and notes for the EDA and preprocessing sections.
- `KAGGLE_ATTEMPTS.md`: model comparisons, attempt history, and final-model findings.

The two Markdown files are supporting notes, not substitutes for the report.
The report includes the confirmed Kaggle team and usernames, plus the project plan and
statement of work division after the five assessed pages. Neal Liu's staff response in forum thread 81 confirms
that no specific template exists and a clear format of our choice is acceptable.
The cover-sheet contribution table does not replace this appendix.

The original PDF received from Aditya in Downloads has not been overwritten.
The source of the numerical evidence is now
`../development/FIT5149_A1_final_clean.ipynb`. Its sections follow the report and
recalculate the main tables and Figure 1. The generated files and provenance record
are in `evidence/`. Kaggle scores are labelled as external observations, not local estimates.

`../development/build_report.py` is only the earlier layout builder. It reads the
notebook's exported tables and figure and writes a separate five-page rebuild draft.
That draft does not include the manually edited appendix. Do not replace the current
report with it without reviewing and restoring those edits.
Export and visually check the PDF again after any Word changes.

Use `final_clean` to follow the report calculations. Use the root final notebook for
the standalone attempt 4 implementation. Attempt 5 is a later, unselected experiment.
The Word templates are in `../forms/`.
