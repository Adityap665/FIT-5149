"""Run the delivery notebook in a fresh kernel and compare its prediction file."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent


def verify(notebook_name, prediction_name):
    notebook_path = ROOT / notebook_name
    reference = ROOT / prediction_name
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    notebook = nbformat.read(notebook_path, as_version=4)
    original_count = len(notebook.cells)
    code_hash = hashlib.sha256("\n".join(c.source for c in notebook.cells).encode()).hexdigest()

    # Keep this review export out of the delivered notebook.
    notebook.cells.append(nbformat.v4.new_code_cell('''import json
review = {"model": MODEL_NAME, "validation": validation_summary.to_dict()}
review["permutation_checks"] = permutation_summary.reset_index().to_dict(orient="records")
review["error_groups"] = {}
for column in ["approved", "credit_missing", "request_band", "sanction_rate"]:
    table = diagnostics.groupby(column, observed=True).apply(error_summary, include_groups=False).reset_index()
    table[column] = table[column].astype(str)
    review["error_groups"][column] = table.to_dict(orient="records")
(DATA_DIR / "_verification.json").write_text(json.dumps(review, indent=2, default=str))
'''))

    with tempfile.TemporaryDirectory(prefix="fit5149-final-check-") as scratch:
        scratch_path = Path(scratch)
        for name in ("training_set.csv", "kaggle_test_X.csv", "sample_submission.csv"):
            shutil.copy2(ROOT / name, scratch_path / name)
        os.environ["MPLCONFIGDIR"] = str(scratch_path / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(scratch_path / "cache")
        os.environ["IPYTHONDIR"] = str(scratch_path / "ipython")
        os.environ["JUPYTER_RUNTIME_DIR"] = str(scratch_path / "jupyter-runtime")
        os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

        def report_cell(cell, cell_index, **kwargs):
            if cell.cell_type == "code" and cell_index < original_count:
                print(f"Executed code cell {cell_index + 1}/{original_count}", flush=True)

        client = NotebookClient(
            notebook, kernel_name="fit5196", timeout=600,
            resources={"metadata": {"path": scratch}},
            on_cell_executed=report_cell,
        )
        client.execute()
        generated = scratch_path / "submission.csv"
        generated_hash = hashlib.sha256(generated.read_bytes()).hexdigest()
        if generated_hash != reference_hash:
            raise AssertionError("The notebook did not reproduce the candidate CSV byte for byte.")
        record = json.loads((scratch_path / "_verification.json").read_text())
        record.update(notebook=notebook_name, candidate_file=prediction_name,
                      code_sha256=code_hash, prediction_sha256=generated_hash,
                      fresh_kernel_passed=True, exact_csv_match=True)
        (ROOT / "model_checks" / "final_verification.json").write_text(json.dumps(record, indent=2) + "\n")
        destination = ROOT / "submission.csv"
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != generated_hash:
            raise FileExistsError("An existing, different submission.csv must be preserved.")
        shutil.copy2(generated, destination)
        notebook.cells = notebook.cells[:original_count]
        nbformat.write(notebook, notebook_path)
        print("PASS: fresh kernel, all notebook cells, format checks, and exact CSV reproduction.", flush=True)
        print(json.dumps(record["validation"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    parser.add_argument("prediction")
    arguments = parser.parse_args()
    verify(arguments.notebook, arguments.prediction)
