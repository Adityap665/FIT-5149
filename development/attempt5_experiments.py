"""Compare a small set of two-stage variants without changing attempt 4.

Run from the repository root: python development/attempt5_experiments.py
"""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold

import loan_experiments as earlier

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "development" / "model_checks"
CONFIGS = {
    "attempt4_reference": dict(seeds=[42], approval_iterations=1200, weight_power=0),
    "seed_average": dict(seeds=[42, 73, 2026], approval_iterations=1200, weight_power=0),
    "request_weighted": dict(seeds=[42], approval_iterations=1200, weight_power=1),
    "shorter_approval": dict(seeds=[42], approval_iterations=600, weight_power=0),
    "squared_approval": dict(seeds=[42], approval_iterations=1200, weight_power=0,
                             approval_objective="squared", approval_depth=5),
    "squared_approval_shallow": dict(seeds=[42], approval_iterations=1200, weight_power=0,
                                     approval_objective="squared", approval_depth=3),
}


def fit_candidate(features, amounts, categories, config):
    # All fitting and weighting use this training fold only.
    requested = features["requested_amount_aud"].to_numpy()
    rates = (np.asarray(amounts) / requested).round(2)
    approved = np.asarray(amounts) > 0
    members = []
    for seed in config["seeds"]:
        shared = dict(learning_rate=0.03, l2_leaf_reg=10.0, random_seed=seed,
                      thread_count=6, verbose=False, allow_writing_files=False)
        approval_options, rate_options = {}, {}
        if config["weight_power"]:
            weights = requested ** config["weight_power"]
            approval_options["sample_weight"] = weights / weights.mean()
            rate_options["sample_weight"] = weights[approved] / weights[approved].mean()
        if config.get("approval_objective") == "squared":
            approval = CatBoostRegressor(loss_function="RMSE", depth=config["approval_depth"],
                                        iterations=config["approval_iterations"], **shared)
        else:
            approval = CatBoostClassifier(loss_function="Logloss", depth=5,
                                         iterations=config["approval_iterations"], **shared)
        approval.fit(features, approved.astype(int), cat_features=categories, **approval_options)
        rate = CatBoostRegressor(loss_function="RMSE", depth=4, iterations=800, **shared)
        rate.fit(features.loc[approved], rates[approved], cat_features=categories, **rate_options)
        members.append((approval, rate))
    return members


def predict_candidate(members, features):
    # Average the complete dollar predictions, not the two stages separately.
    requested = features["requested_amount_aud"].to_numpy()
    predictions = []
    for approval, rate in members:
        if isinstance(approval, CatBoostClassifier):
            positive_column = list(approval.classes_).index(1)
            probability = approval.predict_proba(features)[:, positive_column]
        else:
            probability = np.clip(approval.predict(features), 0, 1)
        positive_rate = np.clip(rate.predict(features), 0, 1)
        predictions.append(requested * probability * positive_rate)
    return np.clip(np.mean(predictions, axis=0), 0, requested)


def model_code_hash():
    source = inspect.getsource(fit_candidate) + inspect.getsource(predict_candidate)
    return hashlib.sha256(source.encode()).hexdigest()


def evaluate(names, split_seed):
    train, test, sample, features, test_features, categories = earlier.load_data()
    y = train[earlier.TARGET]
    strata = (y / train["requested_amount_aud"]).round(2).map(lambda rate: f"{rate:.2f}")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=split_seed).split(features, strata))
    cache = RESULTS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in names:
        config = CONFIGS[name]
        result_path = RESULTS / f"attempt5_{name}_seed{split_seed}.json"
        cache_path = cache / f"attempt5_{name}_seed{split_seed}.npz"
        identity = dict(config=config, split_seed=split_seed, data_sha256=earlier.data_fingerprint(),
                        model_code_sha256=model_code_hash(), features=features.columns.tolist())
        if result_path.exists() and cache_path.exists():
            previous = json.loads(result_path.read_text())
            if all(previous.get(key) == value for key, value in identity.items()):
                summaries.append(previous)
                print(f"Cached: {name}, split {split_seed}", flush=True)
                continue
        started = time.time()
        oof = np.full(len(train), np.nan)
        rows = []
        for fold, (fitting, validation) in enumerate(folds, 1):
            members = fit_candidate(features.iloc[fitting], y.iloc[fitting], categories, config)
            predictions = predict_candidate(members, features.iloc[validation])
            oof[validation] = predictions
            row = dict(fold=fold, rmse_aud=earlier.rmse(y.iloc[validation], predictions),
                       mae_aud=float(mean_absolute_error(y.iloc[validation], predictions)))
            rows.append(row)
            print(f"{name}, split {split_seed}, fold {fold}: RMSE {row['rmse_aud']:,.2f}", flush=True)
        assert np.isfinite(oof).all()
        scores = np.array([row["rmse_aud"] for row in rows])
        record = dict(identity, name=name, mean_fold_rmse_aud=float(scores.mean()),
                      global_oof_rmse_aud=earlier.rmse(y, oof),
                      sd_fold_rmse_aud=float(scores.std(ddof=1)),
                      mae_aud=float(mean_absolute_error(y, oof)), folds=rows,
                      seconds=time.time()-started)
        result_path.write_text(json.dumps(record, indent=2) + "\n")
        np.savez_compressed(cache_path, prediction=oof)
        summaries.append(record)
        print(f"DONE {name}: mean RMSE {scores.mean():,.2f}", flush=True)
    print(pd.DataFrame([{key: result[key] for key in
          ["name", "split_seed", "mean_fold_rmse_aud", "global_oof_rmse_aud", "mae_aud"]}
          for result in summaries]).sort_values("mean_fold_rmse_aud").to_string(index=False), flush=True)


def export(name, filename):
    if Path(filename).name != filename or not filename.endswith(".csv"):
        raise ValueError("Provide a CSV filename without folders.")
    destination = ROOT / "submissions" / filename
    if destination.exists():
        raise FileExistsError(f"Keep the existing file: {destination}")
    train, test, sample, features, test_features, categories = earlier.load_data()
    members = fit_candidate(features, train[earlier.TARGET], categories, CONFIGS[name])
    predictions = predict_candidate(members, test_features).round(2)
    result = sample[["application_id"]].copy()
    result[earlier.TARGET] = predictions
    assert result.shape == (10000, 2) and result.columns.tolist() == sample.columns.tolist()
    assert result["application_id"].equals(test["application_id"])
    assert np.isfinite(predictions).all()
    assert result[earlier.TARGET].between(0, test["requested_amount_aud"]).all()
    result.to_csv(destination, index=False)
    record = dict(model=name, config=CONFIGS[name], model_code_sha256=model_code_hash(),
                  data_sha256=earlier.data_fingerprint(), rows=len(result),
                  file=str(destination.relative_to(ROOT)),
                  prediction_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    (RESULTS / "attempt5_export.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


def prepare_notebook(name, filename):
    """Keep a standalone implementation of the new attempt, separate from attempt 4."""
    import nbformat

    destination = ROOT / "development" / "FIT5149_A1_attempt5.ipynb"
    if destination.exists():
        raise FileExistsError(f"Keep the existing notebook: {destination}")
    notebook = nbformat.read(ROOT / "FIT5149_A1_final.ipynb", as_version=4)
    explanation = {
        "seed_average": "We train the same two-stage CatBoost model three times, with seeds 42, 73, and 2026. We average the three complete dollar predictions with equal weights. This reduces dependence on one random training run without choosing weights from the Kaggle leaderboard.",
        "request_weighted": "We keep the two-stage model but give larger requests more weight during fitting. Weights are proportional to the amount requested and normalised using the training fold only. This is a partial adjustment toward the dollar-based error, not an exact optimisation of dollar RMSE.",
        "shorter_approval": "We keep the same two-stage model but reduce the approval classifier from 1,200 to 600 iterations. The positive-rate model is unchanged. The aim is to reduce overfitting in the approval step.",
        "attempt4_reference": "This is a reproduction of attempt 4 and is not a new Kaggle candidate.",
        "squared_approval": "We predict the binary approval outcome with squared-error regression and limit its output to a probability between zero and one. The positive-rate model is unchanged. This tests a different probability-fitting objective while keeping the same data and validation folds.",
        "squared_approval_shallow": "We predict approval with a shallow squared-error regression model and limit its output to a probability between zero and one. The positive-rate model is unchanged. This tests a different probability-fitting objective with less complex trees.",
    }[name]
    notebook.cells[0].source = "# FIT5149 Assessment 1: Kaggle attempt 5\n\nGroup 43.\n\n" + explanation + "\n\nThis candidate is kept separately. The root final notebook still reproduces attempt 4 until the team makes its final selection."
    notebook.cells[2].source = "## 1. Load the supplied data\n\nThe three input CSVs are in the repository root, one folder above this notebook. It can also run from the root folder."
    notebook.cells[3].source = notebook.cells[3].source.replace('Path.cwd() / "Assignment" / "A1"', 'Path.cwd().parent')
    notebook.cells[12].source = "## 6. Candidate for attempt 5\n\n" + explanation
    notebook.cells[13].source = (
        f"MODEL_NAME = {name!r}\nCANDIDATE_CONFIG = {CONFIGS[name]!r}\n\n"
        + inspect.getsource(fit_candidate) + "\n\n" + inspect.getsource(predict_candidate)
        + "\n\ndef fit_final_model(features, amounts):\n"
        + "    return fit_candidate(features, amounts, selected_categorical_features, CANDIDATE_CONFIG)\n\n"
        + "def predict_final_model(models, features):\n"
        + "    return predict_candidate(models, features)\n"
    )
    # Do not carry attempt 4's numeric interpretations into a different model.
    notebook.cells[18].source = (
        "## 9. Train on all labelled data and create attempt 5\n\n"
        "The tables above describe this candidate's own validation errors. We compare errors for "
        "approved and rejected applications, missing credit ratings, and request sizes. These are "
        "out-of-fold development results, not an independent test.\n\n"
        "We now fit the candidate on all labelled rows and save a separate CSV under submissions/. "
        "The attempt 4 files are not replaced. The public Kaggle result remains unknown until upload."
    )
    notebook.cells[19].source = notebook.cells[19].source.replace(
        'submission_path = DATA_DIR / "submission.csv"',
        f'submission_path = DATA_DIR / "submissions" / {filename!r}\nsubmission_path.parent.mkdir(exist_ok=True)'
    )
    notebook.cells[21].source = notebook.cells[21].source.replace(
        '"final_output": "submission.csv",',
        f'"model_seeds": str(CANDIDATE_CONFIG["seeds"]),\n    "final_output": {filename!r},'
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    nbformat.write(notebook, destination)
    print(f"Created standalone notebook: {destination}", flush=True)


def verify_notebook(filename):
    """Execute the candidate in isolation and require exactly matching predictions."""
    import os
    import shutil
    import tempfile
    import nbformat
    from nbclient import NotebookClient

    notebook_path = ROOT / "development" / "FIT5149_A1_attempt5.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    original_count = len(notebook.cells)
    notebook.cells.append(nbformat.v4.new_code_cell(
        'import json\n'
        'review = {"model": MODEL_NAME, "validation": validation_summary.to_dict()}\n'
        'review["permutation_checks"] = permutation_summary.reset_index().to_dict(orient="records")\n'
        'review["error_groups"] = {}\n'
        'for column in ["approved", "credit_missing", "request_band", "sanction_rate"]:\n'
        '    table = diagnostics.groupby(column, observed=True).apply(error_summary, include_groups=False).reset_index()\n'
        '    table[column] = table[column].astype(str)\n'
        '    review["error_groups"][column] = table.to_dict(orient="records")\n'
        '(DATA_DIR / "review.json").write_text(json.dumps(review, indent=2, default=str))\n'
    ))
    with tempfile.TemporaryDirectory(prefix="fit5149-attempt5-") as folder:
        scratch = Path(folder)
        for name in ["training_set.csv", "kaggle_test_X.csv", "sample_submission.csv"]:
            shutil.copy2(ROOT / name, scratch / name)
        env = dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline",
                   MPLCONFIGDIR=str(scratch / "matplotlib"), IPYTHONDIR=str(scratch / "ipython"),
                   JUPYTER_RUNTIME_DIR=str(scratch / "jupyter-runtime"))
        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"Verified execution cell {cell_index + 1}/{original_count}", flush=True)
        client = NotebookClient(notebook, kernel_name="fit5196", timeout=900,
                                resources={"metadata": {"path": str(scratch)}},
                                on_cell_executed=progress)
        client.execute(env=env)
        expected = (ROOT / "submissions" / filename).read_bytes()
        actual = (scratch / "submissions" / filename).read_bytes()
        assert expected == actual, "Fresh notebook predictions differ from the export."
        record = json.loads((scratch / "review.json").read_text())
        record.update(fresh_kernel_passed=True, exact_csv_match=True,
                      prediction_sha256=hashlib.sha256(actual).hexdigest(),
                      notebook=str(notebook_path.relative_to(ROOT)))
    notebook.cells = notebook.cells[:original_count]
    record["notebook_source_sha256"] = hashlib.sha256("\n".join(c.source for c in notebook.cells).encode()).hexdigest()
    nbformat.write(notebook, notebook_path)
    (RESULTS / "attempt5_verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print("PASS: standalone notebook, fresh kernel, exact CSV match.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--export", metavar="CSV_NAME")
    parser.add_argument("--prepare-notebook", metavar="CSV_NAME")
    parser.add_argument("--verify-notebook", metavar="CSV_NAME")
    args = parser.parse_args()
    if args.verify_notebook:
        verify_notebook(args.verify_notebook)
    elif args.prepare_notebook:
        if len(args.models) != 1:
            parser.error("Choose exactly one model for the notebook.")
        prepare_notebook(args.models[0], args.prepare_notebook)
    elif args.export:
        if len(args.models) != 1:
            parser.error("Choose exactly one model to export.")
        export(args.models[0], args.export)
    else:
        evaluate(args.models, args.split_seed)
