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
The report source builder is `../development/build_report.py`. It uses saved results
and does not train or change models. Manual Word edits should be kept separately
or carried into the builder before it is rerun, because rerunning replaces the Word file
and removes the manually merged appendix. Do not rerun it over the current report.
Export and visually check the PDF again after any Word changes.

Use the current final notebook for final-model results, not the older clean notebook.
The Word templates are in `../forms/`.
