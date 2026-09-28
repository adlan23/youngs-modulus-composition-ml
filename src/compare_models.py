"""Evaluate fixed baseline models on the same records as the XGBoost model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import create_xgb_model, gpu_is_available, prepare_features, sample_weights


RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory containing mp_features.csv and composition_groups.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which comparison results will be written.",
    )
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def metrics(observed: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "R2": r2_score(observed, predicted),
        "RMSE_log10E": float(np.sqrt(mean_squared_error(observed, predicted))),
        "MAE_log10E": mean_absolute_error(observed, predicted),
    }


def evaluate_partition(
    split_name: str,
    training_indices: np.ndarray,
    test_indices: np.ndarray,
    predictors_raw: pd.DataFrame,
    target: pd.Series,
    use_gpu: bool,
) -> list[dict[str, object]]:
    raw_train = predictors_raw.iloc[training_indices]
    raw_test = predictors_raw.iloc[test_indices]
    y_train = target.iloc[training_indices]
    y_test = target.iloc[test_indices]
    x_train, x_test = prepare_features(raw_train, raw_test)
    weights = sample_weights(y_train)

    models = [
        ("Training median", DummyRegressor(strategy="median")),
        ("Ridge", Ridge(alpha=1.0)),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                max_features=0.8,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        ),
        ("XGBoost", create_xgb_model(use_gpu, random_state=RANDOM_SEED)),
    ]

    records: list[dict[str, object]] = []
    for model_name, model in models:
        if model_name == "Ridge":
            scaler = StandardScaler().fit(x_train, sample_weight=weights)
            scaled_train = scaler.transform(x_train)
            scaled_test = scaler.transform(x_test)
            model.fit(scaled_train, y_train, sample_weight=weights)
            prediction = model.predict(scaled_test)
        else:
            model.fit(x_train, y_train, sample_weight=weights)
            prediction = model.predict(x_test)

        records.append(
            {
                "Split": split_name,
                "Model": model_name,
                "Train_samples": len(training_indices),
                "Test_samples": len(test_indices),
                "Input_features": x_train.shape[1],
                **metrics(y_test, prediction),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    feature_path = args.processed_dir / "mp_features.csv"
    group_path = args.processed_dir / "composition_groups.csv"
    for path in (feature_path, group_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    frame = pd.read_csv(feature_path)
    formulae = pd.read_csv(group_path)["reduced_formula"]
    target = frame["logE"]
    predictors_raw = frame.drop(columns=["logE", "E", "K", "G"], errors="ignore")

    if len(frame) != len(formulae):
        raise ValueError("Feature and formula metadata rows are not aligned.")
    if not all(
        name.startswith("MagpieData ") or re.fullmatch(r"[0-9]+-norm", name)
        for name in predictors_raw.columns
    ):
        raise ValueError("A non-composition input column was detected.")

    row_indices = np.arange(len(target))
    random_train, random_test = train_test_split(
        row_indices, test_size=0.2, random_state=RANDOM_SEED
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    grouped_train, grouped_test = next(
        splitter.split(predictors_raw, target, groups=formulae)
    )
    if set(formulae.iloc[grouped_train]).intersection(formulae.iloc[grouped_test]):
        raise RuntimeError("Reduced-formula leakage was detected.")

    use_gpu = gpu_is_available(args.cpu_only)
    records = evaluate_partition(
        "Random 80:20",
        random_train,
        random_test,
        predictors_raw,
        target,
        use_gpu,
    )
    records += evaluate_partition(
        "Reduced-formula grouped",
        grouped_train,
        grouped_test,
        predictors_raw,
        target,
        use_gpu,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(records)
    comparison.to_csv(args.output_dir / "same_data_model_comparison.csv", index=False)
    diagnostics = pd.DataFrame(
        {
            "Split": ["Random 80:20", "Reduced-formula grouped"],
            "Train_formulae": [
                formulae.iloc[random_train].nunique(),
                formulae.iloc[grouped_train].nunique(),
            ],
            "Test_formulae": [
                formulae.iloc[random_test].nunique(),
                formulae.iloc[grouped_test].nunique(),
            ],
            "Formula_overlap": [
                len(
                    set(formulae.iloc[random_train]).intersection(
                        formulae.iloc[random_test]
                    )
                ),
                0,
            ],
        }
    )
    diagnostics.to_csv(args.output_dir / "split_formula_diagnostics.csv", index=False)

    design = {
        "target": "log10(E in GPa)",
        "weighted_training": "1 / (abs(y_train - mean(y_train)) + 0.1)",
        "splits": [
            "train_test_split 80:20, random_state=42",
            "GroupShuffleSplit by reduced formula, random_state=42",
        ],
        "preprocessing": (
            "Fit on training rows only; retain >=70% observed columns, median "
            "imputation, remove one of each pair with abs(Pearson r) > 0.95"
        ),
        "models": {
            "Training median": "DummyRegressor(strategy='median')",
            "Ridge": "weighted StandardScaler + Ridge(alpha=1.0)",
            "Random Forest": (
                "300 trees, min_samples_leaf=2, max_features=0.8, random_state=42"
            ),
            "XGBoost": "fixed parameters reported in the manuscript",
        },
        "gpu_used": use_gpu,
        "note": "Fixed baselines; no hyperparameter search was performed.",
    }
    with (args.output_dir / "comparison_design.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(design, handle, indent=2)

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
