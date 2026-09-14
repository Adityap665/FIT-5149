"""Small, reproducible model comparison for the last Kaggle attempts.

Run from this folder with the Python environment listed in requirements.txt.
This is development code. The final delivery notebook contains only its chosen model.
"""

from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parent
TARGET = "sanctioned_amount_aud"
SEED = 42
THREADS = 6
SENTINEL_COLUMNS = [
    "existing_repayments_aud", "has_co_applicant", "property_value_aud",
]

# These settings are fixed before looking at the new validation results.
CONFIGS = {
    "catboost_current": dict(kind="rate", depth=7, iterations=1200, learning_rate=0.03, l2_leaf_reg=5.0),
    "catboost_shallow": dict(kind="rate", depth=4, iterations=1500, learning_rate=0.03, l2_leaf_reg=10.0),
    "catboost_weighted": dict(kind="rate", depth=4, iterations=1500, learning_rate=0.03, l2_leaf_reg=10.0, weight_power=2),
    "catboost_amount": dict(kind="amount", depth=6, iterations=1200, learning_rate=0.03, l2_leaf_reg=10.0),
    "catboost_two_stage": dict(kind="two_stage", depth=5, iterations=1200, learning_rate=0.03, l2_leaf_reg=10.0),
    "catboost_tiers": dict(kind="tiers", depth=4, iterations=1200, learning_rate=0.03, l2_leaf_reg=10.0),
    # Check whether the earlier feature evidence helps remove noisy inputs.
    "catboost_core": dict(kind="rate", depth=5, iterations=1000, learning_rate=0.03, l2_leaf_reg=10.0, feature_set="core"),
    "catboost_two_stage_core": dict(kind="two_stage", depth=5, iterations=1200, learning_rate=0.03, l2_leaf_reg=10.0, feature_set="core"),
    "catboost_two_stage_small": dict(kind="two_stage", depth=4, iterations=800, learning_rate=0.03, l2_leaf_reg=10.0, feature_set="small"),
    "xgboost_rate": dict(kind="xgb_rate", depth=4, iterations=800, learning_rate=0.03, l2_leaf_reg=10.0),
    "xgboost_two_stage": dict(kind="xgb_two_stage", depth=4, iterations=800, learning_rate=0.03, l2_leaf_reg=10.0),
    "catboost_two_stage_deeper": dict(kind="two_stage", depth=7, iterations=1200, learning_rate=0.03, l2_leaf_reg=5.0),
    "catboost_two_stage_longer": dict(kind="two_stage", depth=6, iterations=1600, learning_rate=0.03, l2_leaf_reg=3.0),
    "catboost_two_stage_smooth": dict(kind="two_stage", depth=3, iterations=1000, learning_rate=0.03, l2_leaf_reg=20.0),
    "catboost_rate_longer": dict(kind="rate", depth=6, iterations=2500, learning_rate=0.03, l2_leaf_reg=10.0),
}


def rmse(actual, predicted):
    return float(mean_squared_error(actual, predicted) ** 0.5)


def engineer_features(frame, missing_columns):
    """Use the same row-by-row transformations as Aditya's final notebook."""
    x = frame.copy()
    for column in SENTINEL_COLUMNS:
        x[column] = x[column].replace(-999, np.nan)

    x["missing_value_count"] = x.isna().sum(axis=1)
    for column in missing_columns:
        x[f"{column}_is_missing"] = x[column].isna().astype("int8")

    date = pd.to_datetime(x.pop("application_date"), format="%d/%m/%Y", errors="raise")
    x["application_year"] = date.dt.year
    x["application_month_sin"] = np.sin(2 * np.pi * date.dt.month / 12)
    x["application_month_cos"] = np.cos(2 * np.pi * date.dt.month / 12)
    x["application_dayofyear_sin"] = np.sin(2 * np.pi * date.dt.dayofyear / 365.25)
    x["application_dayofyear_cos"] = np.cos(2 * np.pi * date.dt.dayofyear / 365.25)

    incomes = pd.concat([x["annual_income_aud"], 12 * x["monthly_income_aud"]], axis=1)
    x["annual_income_reconciled_aud"] = incomes.median(axis=1, skipna=True)
    x["income_sources_available"] = incomes.notna().sum(axis=1)
    x["income_relative_gap"] = (
        (x["annual_income_aud"] - 12 * x["monthly_income_aud"]).abs()
        / x["annual_income_reconciled_aud"].replace(0, np.nan)
    )
    x["requested_to_property_value"] = x["requested_amount_aud"] / x["property_value_aud"].replace(0, np.nan)
    x["requested_to_annual_income"] = x["requested_amount_aud"] / x["annual_income_reconciled_aud"].replace(0, np.nan)
    x["repayments_to_monthly_income"] = x["existing_repayments_aud"] / (x["annual_income_reconciled_aud"] / 12).replace(0, np.nan)
    x["property_value_minus_requested"] = x["property_value_aud"] - x["requested_amount_aud"]
    return x.drop(columns=["application_id", "property_age_years", "property_ref"])


def load_data():
    train = pd.read_csv(ROOT / "training_set.csv")
    test = pd.read_csv(ROOT / "kaggle_test_X.csv")
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    assert train.shape == (19322, 29) and test.shape == (10000, 28)
    assert train.drop(columns=TARGET).columns.tolist() == test.columns.tolist()
    assert train["application_id"].is_unique and test["application_id"].is_unique
    assert not train["application_id"].isin(test["application_id"]).any()
    assert sample["application_id"].equals(test["application_id"])
    assert sample.columns.tolist() == ["application_id", TARGET]
    assert (train["requested_amount_aud"] > 0).all()
    assert (test["requested_amount_aud"] > 0).all()

    missing_columns = [c for c in test.columns if train[c].isna().any() or c in SENTINEL_COLUMNS]
    x = engineer_features(train.drop(columns=TARGET), missing_columns)
    x_test = engineer_features(test, missing_columns)
    categories = x.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for c in categories:
        x[c] = x[c].astype("string").fillna("__MISSING__").astype(str)
        x_test[c] = x_test[c].astype("string").fillna("__MISSING__").astype(str)
    assert x.columns.tolist() == x_test.columns.tolist()
    for frame in (x, x_test):
        assert not np.isinf(frame.select_dtypes(include="number").to_numpy()).any()
    return train, test, sample, x, x_test, categories


def model_features(x, config):
    if config.get("feature_set") == "core":
        return x[["credit_rating", "has_co_applicant", "income_consistency", "applicant_age"]]
    if config.get("feature_set") == "small":
        return x[[
            "credit_rating", "has_co_applicant", "income_consistency", "applicant_age",
            "prior_default_count", "risk_review_flag", "occupation_class",
            "annual_income_reconciled_aud", "requested_amount_aud",
            "requested_to_annual_income", "repayments_to_monthly_income",
            "residence_area",
        ]]
    return x


def fit_model(x, y, categories, config):
    params = {k: v for k, v in config.items() if k not in {"kind", "weight_power", "feature_set"}}
    params.update(random_seed=SEED, thread_count=THREADS, verbose=False, allow_writing_files=False)
    requested = x["requested_amount_aud"].to_numpy()
    rate = (np.asarray(y) / requested).round(2)
    kind = config["kind"]
    x = model_features(x, config)
    categories = [c for c in categories if c in x.columns]

    if kind.startswith("xgb_"):
        from xgboost import XGBClassifier, XGBRegressor
        # Learn the category vocabulary from this training fold only.
        levels = {c: sorted(x[c].unique().tolist()) for c in categories}
        encoded = encode_xgboost(x, levels)
        xgb_params = dict(
            n_estimators=config["iterations"], max_depth=config["depth"],
            learning_rate=config["learning_rate"], reg_lambda=config["l2_leaf_reg"],
            min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", enable_categorical=True,
            random_state=SEED, n_jobs=THREADS,
        )
        if kind == "xgb_two_stage":
            approved = np.asarray(y) > 0
            approval_model = XGBClassifier(objective="binary:logistic", **xgb_params)
            approval_model.fit(encoded, approved.astype(int))
            rate_model = XGBRegressor(objective="reg:squarederror", **xgb_params)
            rate_model.fit(encoded.loc[approved], rate[approved])
            return dict(levels=levels, approval=approval_model, rate=rate_model)
        model = XGBRegressor(objective="reg:squarederror", **xgb_params)
        model.fit(encoded, rate)
        return dict(levels=levels, rate=model)

    if kind == "two_stage":
        approved = np.asarray(y) > 0
        approval_model = CatBoostClassifier(loss_function="Logloss", **params)
        approval_model.fit(x, approved.astype(int), cat_features=categories)
        positive_params = dict(params, depth=4, iterations=800)
        rate_model = CatBoostRegressor(loss_function="RMSE", **positive_params)
        rate_model.fit(x.loc[approved], rate[approved], cat_features=categories)
        return approval_model, rate_model

    if kind == "tiers":
        model = CatBoostClassifier(loss_function="MultiClass", **params)
        model.fit(x, np.rint(rate * 100).astype(int), cat_features=categories)
        return model

    model = CatBoostRegressor(loss_function="RMSE", **params)
    fit_options = {}
    if config.get("weight_power"):
        # A rate error's squared dollar cost scales with the request squared.
        weights = requested ** config["weight_power"]
        fit_options["sample_weight"] = weights / weights.mean()
    target = np.asarray(y) if kind == "amount" else rate
    model.fit(x, target, cat_features=categories, **fit_options)
    return model


def encode_xgboost(x, levels):
    encoded = x.copy()
    for c, values in levels.items():
        known = encoded[c].where(encoded[c].isin(values))
        encoded[c] = pd.Categorical(known, categories=values)
    return encoded


def predict_amount(model, x, config):
    requested = x["requested_amount_aud"].to_numpy()
    kind = config["kind"]
    x = model_features(x, config)
    if kind.startswith("xgb_"):
        encoded = encode_xgboost(x, model["levels"])
        rate = np.clip(model["rate"].predict(encoded), 0, 1)
        if kind == "xgb_two_stage":
            rate = rate * model["approval"].predict_proba(encoded)[:, 1]
        amount = requested * rate
    elif kind == "two_stage":
        approval_model, rate_model = model
        positive_column = list(approval_model.classes_).index(1)
        approval_probability = approval_model.predict_proba(x)[:, positive_column]
        positive_rate = np.clip(rate_model.predict(x), 0, 1)
        amount = requested * approval_probability * positive_rate
    elif kind == "tiers":
        # Retain uncertainty instead of choosing a single approval tier.
        rate = model.predict_proba(x) @ (model.classes_.astype(float) / 100)
        amount = requested * rate
    elif kind == "amount":
        amount = model.predict(x)
    else:
        amount = requested * np.clip(model.predict(x), 0, 1)
    return np.clip(amount, 0, requested)


def data_fingerprint():
    digest = hashlib.sha256()
    for name in ("training_set.csv", "kaggle_test_X.csv", "sample_submission.csv"):
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def evaluate(names, split_seed=42):
    train, test, sample, x, x_test, categories = load_data()
    y = train[TARGET]
    strata = (y / train["requested_amount_aud"]).round(2).map(lambda v: f"{v:.2f}")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed).split(x, strata))
    out = ROOT / "model_checks"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in names:
        config = CONFIGS[name]
        result_path = out / f"{name}_seed{split_seed}.json"
        cache_path = cache / f"{name}_seed{split_seed}.npz"
        identity = dict(config=config, split_seed=split_seed, model_seed=SEED,
                        data_sha256=data_fingerprint(), features=x.columns.tolist())
        if result_path.exists() and cache_path.exists():
            previous = json.loads(result_path.read_text())
            if all(previous.get(k) == v for k, v in identity.items()):
                print(f"Already evaluated: {name}, seed {split_seed}", flush=True)
                summaries.append(previous)
                continue
        predictions = np.full(len(train), np.nan)
        rows = []
        started = time.time()
        for fold, (fitting, validation) in enumerate(folds, 1):
            fold_started = time.time()
            model = fit_model(x.iloc[fitting], y.iloc[fitting], categories, config)
            predicted = predict_amount(model, x.iloc[validation], config)
            predictions[validation] = predicted
            row = dict(fold=fold, rmse_aud=rmse(y.iloc[validation], predicted),
                       mae_aud=float(mean_absolute_error(y.iloc[validation], predicted)))
            rows.append(row)
            print(f"{name}: fold {fold}/5, RMSE {row['rmse_aud']:,.2f}, {time.time()-fold_started:.1f}s", flush=True)
        assert np.isfinite(predictions).all()
        fold_rmse = np.array([r["rmse_aud"] for r in rows])
        summary = dict(identity, name=name, mean_fold_rmse_aud=float(fold_rmse.mean()),
                       sd_fold_rmse_aud=float(fold_rmse.std(ddof=1)),
                       global_oof_rmse_aud=rmse(y, predictions),
                       mae_aud=float(mean_absolute_error(y, predictions)), folds=rows,
                       seconds=time.time()-started)
        np.savez_compressed(cache_path, prediction=predictions)
        result_path.write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(f"DONE {name}: mean-fold RMSE {summary['mean_fold_rmse_aud']:,.2f}; global OOF {summary['global_oof_rmse_aud']:,.2f}", flush=True)
    table = pd.DataFrame([{k: s[k] for k in ["name", "split_seed", "mean_fold_rmse_aud", "global_oof_rmse_aud", "mae_aud", "seconds"]} for s in summaries])
    print(table.sort_values("global_oof_rmse_aud").to_string(index=False), flush=True)


def export_prediction(name, output_name):
    if Path(output_name).name != output_name or not output_name.endswith(".csv"):
        raise ValueError("Use a CSV filename in the repository folder.")
    destination = ROOT / output_name
    if destination.exists():
        raise FileExistsError(f"Keep the existing prediction file: {destination.name}")
    train, test, sample, x, x_test, categories = load_data()
    config = CONFIGS[name]
    started = time.time()
    model = fit_model(x, train[TARGET], categories, config)
    submission = sample[["application_id"]].copy()
    submission[TARGET] = predict_amount(model, x_test, config).round(2)
    assert submission.shape == (10000, 2)
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission["application_id"].equals(test["application_id"])
    assert np.isfinite(submission[TARGET]).all()
    assert submission[TARGET].between(0, test["requested_amount_aud"]).all()
    submission.to_csv(destination, index=False)
    print(f"Created {destination.name}, 10,000 rows, SHA256 {hashlib.sha256(destination.read_bytes()).hexdigest()}, {time.time()-started:.1f}s", flush=True)
    print(submission.head().to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--export", metavar="CSV_NAME")
    args = parser.parse_args()
    if args.export:
        if len(args.models) != 1:
            parser.error("Choose exactly one model when exporting a prediction file.")
        export_prediction(args.models[0], args.export)
    else:
        evaluate(args.models, args.split_seed)
