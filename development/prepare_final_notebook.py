"""Build a standalone final notebook from the selected, validated candidate.

The generated notebook includes only the chosen model. It does not depend on this
script or on the development experiment runner.
"""

import argparse
import copy
import json
from pathlib import Path
import pprint

import nbformat

import loan_experiments as experiments


ROOT = Path(__file__).resolve().parent.parent


def build_notebook(model_name, filename):
    config = experiments.CONFIGS[model_name]
    if config["kind"] not in {"rate", "two_stage"}:
        raise ValueError("This final notebook builder supports the selected CatBoost candidates.")
    if config.get("weight_power"):
        raise ValueError("The weighted candidate was rejected and is not a supported final model.")
    source = nbformat.read(ROOT / "development" / "FIT5149_A1_final_clean.ipynb", as_version=4)
    train, test, sample, x, xt, categories = experiments.load_data()
    predictors = experiments.model_features(x, config).columns.tolist()
    params = {k: v for k, v in config.items() if k not in {"kind", "feature_set", "weight_power"}}
    params.update(random_seed=42, thread_count=experiments.THREADS, verbose=False, allow_writing_files=False)
    two_stage = config["kind"] == "two_stage"
    title = "CatBoost approval probability and sanction rate" if two_stage else "CatBoost sanction-rate regression"
    explanation = (
        "We first estimate the probability that an application receives a positive amount. "
        "A second model learns the sanction rate using approved applications only. "
        "The final amount is the probability of approval multiplied by the expected positive "
        "rate and the amount requested. We keep probabilities rather than using a hard "
        "approve/reject cutoff, because uncertain cases still contribute to squared error."
        if two_stage else
        "The model predicts the share of the requested amount that is approved. We multiply "
        "this predicted rate by the requested amount to obtain a prediction in AUD."
    )
    cells = [nbformat.v4.new_markdown_cell(
        f"# FIT5149 Assessment 1: Final Model Implementation\n\n"
        f"Group 43. Selected model: **{title}**.\n\n{explanation}\n\n"
        "The model was selected using internal cross-validation. Earlier models and the "
        "comparison results are kept in our development files and discussed in the report. "
        "This notebook contains only the selected implementation and its validation."
    )]
    for index in range(1, 12):
        cell = copy.deepcopy(source.cells[index])
        if index == 2:
            cell.source = "## 1. Load the supplied data\n\nKeep the three input CSVs in the same folder as this final notebook."
        if index == 1:
            cell.source = cell.source.replace(
                "from catboost import CatBoostRegressor",
                "from catboost import CatBoostClassifier, CatBoostRegressor" if two_stage
                else "from catboost import CatBoostRegressor",
            )
        if index == 6:
            cell.source = "## 3. Target formulation\n\n" + explanation + "\n\nThe observed tiers are used for stratification only. They are not model inputs."
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
        cells.append(cell)

    model_source = (
        "# Fixed settings selected during the development comparison\n"
        f"MODEL_NAME = {model_name!r}\n"
        f"MODEL_PARAMS = {pprint.pformat(params, sort_dicts=False)}\n"
        f"PREDICTOR_COLUMNS = {pprint.pformat(predictors)}\n"
        "MODEL_CATEGORIES = [c for c in selected_categorical_features if c in PREDICTOR_COLUMNS]\n\n"
    )
    if two_stage:
        model_source += '''def fit_final_model(features, amounts):
    # Each call sees only the fitting rows supplied by the validation loop.
    requested_amounts = features["requested_amount_aud"].to_numpy()
    rates = (np.asarray(amounts) / requested_amounts).round(2)
    approved = np.asarray(amounts) > 0
    predictors = features[PREDICTOR_COLUMNS]

    approval_model = CatBoostClassifier(loss_function="Logloss", **MODEL_PARAMS)
    approval_model.fit(predictors, approved.astype(int), cat_features=MODEL_CATEGORIES)

    rate_params = dict(MODEL_PARAMS, depth=4, iterations=800)
    rate_model = CatBoostRegressor(loss_function="RMSE", **rate_params)
    rate_model.fit(predictors.loc[approved], rates[approved], cat_features=MODEL_CATEGORIES)
    return approval_model, rate_model


def predict_final_model(models, features):
    approval_model, rate_model = models
    predictors = features[PREDICTOR_COLUMNS]
    positive_column = list(approval_model.classes_).index(1)
    approval_probability = approval_model.predict_proba(predictors)[:, positive_column]
    positive_rate = np.clip(rate_model.predict(predictors), 0, 1)
    requested_amounts = features["requested_amount_aud"].to_numpy()
    prediction = requested_amounts * approval_probability * positive_rate
    return np.clip(prediction, 0, requested_amounts)
'''
    else:
        model_source += '''def fit_final_model(features, amounts):
    requested_amounts = features["requested_amount_aud"].to_numpy()
    rates = (np.asarray(amounts) / requested_amounts).round(2)
    model = CatBoostRegressor(loss_function="RMSE", **MODEL_PARAMS)
    model.fit(features[PREDICTOR_COLUMNS], rates, cat_features=MODEL_CATEGORIES)
    return model


def predict_final_model(model, features):
    rate = np.clip(model.predict(features[PREDICTOR_COLUMNS]), 0, 1)
    requested_amounts = features["requested_amount_aud"].to_numpy()
    return np.clip(rate * requested_amounts, 0, requested_amounts)
'''

    cells.extend([
        nbformat.v4.new_markdown_cell("## 6. Selected model\n\n" + explanation),
        nbformat.v4.new_code_cell(model_source),
        nbformat.v4.new_markdown_cell(
            "## 7. Five-fold validation and feature checks\n\n"
            "We score predictions in AUD. The mean fold RMSE matches the metric used in "
            "our earlier comparisons; the global OOF RMSE is calculated across all validation "
            "predictions together. These two summaries need not be identical.\n\n"
            "We also shuffle five inputs identified in the earlier analysis, using three "
            "repeats per fold. This checks whether the selected model depends on those inputs. "
            "It is not a ranking of every feature, a causal test, or a significance test. "
            "Correlated inputs can share information. Shuffling requested amount affects "
            "both its input value and the conversion of the predicted rate to dollars."
        ),
        nbformat.v4.new_code_cell('''validation_rows = []
permutation_rows = []
final_oof = np.full(len(train), np.nan)
FEATURES_TO_CHECK = [
    "requested_amount_aud", "has_co_applicant", "credit_rating",
    "income_consistency", "applicant_age",
]
started = time.time()

for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
    fitted_model = fit_final_model(X_selected.iloc[train_index], y.iloc[train_index])
    validation_x = X_selected.iloc[valid_index]
    prediction = predict_final_model(fitted_model, validation_x)
    final_oof[valid_index] = prediction
    fold_rmse = rmse(y.iloc[valid_index], prediction)
    validation_rows.append({
        "fold": fold_number,
        "rmse_aud": fold_rmse,
        "mae_aud": mean_absolute_error(y.iloc[valid_index], prediction),
    })
    print(f"Fold {fold_number}: RMSE ${fold_rmse:,.2f}")

    for feature_number, feature in enumerate(FEATURES_TO_CHECK):
        for repeat in range(3):
            shuffled = validation_x.copy()
            rng = np.random.default_rng(RANDOM_STATE + 1000 * fold_number + 10 * feature_number + repeat)
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            shuffled_prediction = predict_final_model(fitted_model, shuffled)
            permutation_rows.append({
                "feature": feature, "fold": fold_number, "repeat": repeat + 1,
                "rmse_increase_aud": rmse(y.iloc[valid_index], shuffled_prediction) - fold_rmse,
            })

assert np.isfinite(final_oof).all()
validation_detail = pd.DataFrame(validation_rows)
validation_summary = pd.Series({
    "mean_fold_rmse_aud": validation_detail["rmse_aud"].mean(),
    "sd_fold_rmse_aud": validation_detail["rmse_aud"].std(),
    "global_oof_rmse_aud": rmse(y, final_oof),
    "mae_aud": mean_absolute_error(y, final_oof),
    "validation_seconds": time.time() - started,
})
permutation_detail = pd.DataFrame(permutation_rows)
permutation_by_fold = permutation_detail.groupby(["feature", "fold"])["rmse_increase_aud"].mean()
permutation_summary = permutation_by_fold.groupby("feature").agg(["mean", "std"]).sort_values("mean", ascending=False)
display(validation_summary)
display(permutation_summary)
'''),
        nbformat.v4.new_markdown_cell(
            "## 8. Where the final model makes mistakes\n\n"
            "These tables use the selected model's own out-of-fold predictions. We compare "
            "rejected and approved applications, missing and observed credit ratings, and "
            "five requested-amount bands. The signed error is actual minus predicted, "
            "so a positive value means underprediction. Small groups need cautious interpretation."
        ),
        nbformat.v4.new_code_cell('''diagnostics = pd.DataFrame({
    "actual": y,
    "prediction": final_oof,
    "requested_amount_aud": train["requested_amount_aud"],
    "sanction_rate": sanction_rate,
    "credit_missing": train["credit_rating"].isna(),
    "approved": y > 0,
})
diagnostics["request_band"] = pd.qcut(diagnostics["requested_amount_aud"], 5)
diagnostics["error"] = diagnostics["actual"] - diagnostics["prediction"]

def error_summary(group):
    return pd.Series({
        "rows": len(group),
        "rmse_aud": rmse(group["actual"], group["prediction"]),
        "mae_aud": mean_absolute_error(group["actual"], group["prediction"]),
        "mean_error_aud": group["error"].mean(),
    })

for group_column in ["approved", "credit_missing", "request_band", "sanction_rate"]:
    print(group_column)
    display(diagnostics.groupby(group_column, observed=True).apply(error_summary, include_groups=False))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(validation_detail["fold"].astype(str), validation_detail["rmse_aud"])
ax.axhline(validation_summary["mean_fold_rmse_aud"], linestyle="--", label="Mean fold RMSE")
ax.set(xlabel="Validation fold", ylabel="RMSE (AUD)", title="Selected model validation")
ax.legend()
plt.tight_layout()
plt.show()
'''),
        nbformat.v4.new_markdown_cell(
            "## 9. Retrain on all labelled data and create submission.csv\n\n"
            "We now fit the same selected model on all labelled rows. The exported predictions "
            "must match the Kaggle entry nominated for the final assessment. All format and "
            "range checks run before saving the file."
        ),
        nbformat.v4.new_code_cell('''final_model = fit_final_model(X_selected, y)
final_prediction = predict_final_model(final_model, X_kaggle_selected).round(2)
submission = sample_submission[["application_id"]].copy()
submission[TARGET] = final_prediction

assert submission.shape == (10000, 2)
assert submission.columns.tolist() == sample_submission.columns.tolist()
assert submission["application_id"].equals(test["application_id"])
assert submission["application_id"].is_unique
assert np.isfinite(submission[TARGET]).all()
assert submission[TARGET].between(0, test["requested_amount_aud"]).all()

submission_path = DATA_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved {submission_path.name}: {len(submission):,} checked predictions.")
display(submission.head())
'''),
        nbformat.v4.new_markdown_cell("## 10. Reproducibility information"),
        copy.deepcopy(source.cells[19]),
    ])
    cells[-1].execution_count = None
    cells[-1].outputs = []
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = dict(name="fit5196", display_name="Python (FIT5196)", language="python")
    nbformat.validate(notebook)
    output = ROOT / filename
    if output.exists():
        raise FileExistsError(f"Keep the existing notebook: {output.name}")
    nbformat.write(notebook, output)
    print(f"Created {output.name} for {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=experiments.CONFIGS)
    parser.add_argument("--filename", default="FIT5149_A1_final.ipynb")
    args = parser.parse_args()
    build_notebook(args.model, args.filename)
