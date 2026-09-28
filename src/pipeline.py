"""Reproduce the composition-based XGBoost analysis reported in the manuscript.

The script expects a Materials Project JSON export containing ``bulk_modulus``,
``shear_modulus``, ``state`` and ``formula_pretty`` fields. All preprocessing
that learns from the data is fitted on training rows only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matminer.featurizers.composition import ElementProperty, Stoichiometry
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, train_test_split
from xgboost import XGBRegressor


RANDOM_SEED = 42
ROBUSTNESS_SEEDS = (0, 10, 20, 42, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path("data/raw/mp_elastic_raw.json"),
        help="Path to the Materials Project JSON export.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs"),
        help="Parent directory for timestamped analysis outputs.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable CUDA even when an XGBoost-compatible GPU is available.",
    )
    return parser.parse_args()


def extract_vrh(value: object) -> float:
    """Extract the Voigt-Reuss-Hill modulus from a dictionary."""
    if isinstance(value, dict):
        return value.get("vrh", np.nan)
    return np.nan


def clean_dataset(raw_path: Path, processed_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the raw export, calculate Young's modulus and apply fixed filters."""
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}. See README.md for data preparation."
        )

    with raw_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)
    frame = pd.DataFrame(raw_data)

    required = {"bulk_modulus", "shear_modulus", "state", "formula_pretty"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Raw JSON is missing required fields: {sorted(missing)}")

    counts: dict[str, int] = {"raw": len(frame)}
    frame["K"] = frame["bulk_modulus"].apply(extract_vrh)
    frame["G"] = frame["shear_modulus"].apply(extract_vrh)

    frame = frame.loc[frame["state"] == "successful"].copy()
    counts["successful"] = len(frame)

    frame = frame.dropna(subset=["K", "G"])
    frame = frame.loc[(frame["K"] > 0) & (frame["G"] > 0)].copy()
    counts["positive_elastic_moduli"] = len(frame)

    frame["E"] = (9 * frame["K"] * frame["G"]) / (3 * frame["K"] + frame["G"])
    frame = frame.loc[frame["E"] > 0].copy()
    frame["logE"] = np.log10(frame["E"])
    frame = frame.loc[(frame["logE"] > 0) & (frame["logE"] < 3.5)].copy()
    counts["target_filtered"] = len(frame)

    clean_path = processed_dir / "mp_elastic_clean.csv"
    frame.to_csv(clean_path, index=False)
    with (processed_dir / "cleaning_report.json").open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2)
    return frame, counts


def generate_descriptors(
    clean_frame: pd.DataFrame, processed_dir: Path
) -> tuple[pd.DataFrame, list[str]]:
    """Generate the fixed Magpie and stoichiometric descriptor set."""
    frame = clean_frame.dropna(subset=["formula_pretty"]).copy()
    frame["composition_obj"] = frame["formula_pretty"].apply(Composition)

    groups = pd.DataFrame(
        {
            "formula_pretty": frame["formula_pretty"].to_numpy(),
            "reduced_formula": frame["composition_obj"].map(
                lambda composition: composition.reduced_formula
            ),
        }
    )
    groups.to_csv(processed_dir / "composition_groups.csv", index=False)

    element_properties = ElementProperty.from_preset("magpie")
    stoichiometry = Stoichiometry()
    frame = element_properties.featurize_dataframe(
        frame, col_id="composition_obj", ignore_errors=True
    )
    frame = stoichiometry.featurize_dataframe(
        frame, col_id="composition_obj", ignore_errors=True
    )

    descriptor_columns = (
        element_properties.feature_labels() + stoichiometry.feature_labels()
    )
    if len(descriptor_columns) != len(set(descriptor_columns)):
        raise RuntimeError("Duplicate descriptor names were produced.")
    if not set(descriptor_columns).issubset(frame.columns):
        raise RuntimeError("One or more expected descriptors were not produced.")

    numeric = frame.loc[:, ["logE"] + descriptor_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric.to_csv(processed_dir / "mp_features.csv", index=False)
    return numeric, descriptor_columns


def prepare_features(
    training_raw: pd.DataFrame,
    evaluation_raw: pd.DataFrame,
    correlation_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit column retention, median imputation and filtering on training rows."""
    retained = training_raw.columns[training_raw.notna().mean() >= 0.70]
    training = training_raw.loc[:, retained].copy()
    evaluation = evaluation_raw.loc[:, retained].copy()

    medians = training.median().fillna(0)
    training = training.fillna(medians)
    evaluation = evaluation.fillna(medians)

    correlations = training.corr().abs()
    upper = correlations.where(
        np.triu(np.ones(correlations.shape), k=1).astype(bool)
    )
    drop_columns = [
        column for column in upper if (upper[column] > correlation_threshold).any()
    ]
    return training.drop(columns=drop_columns), evaluation.drop(columns=drop_columns)


def create_xgb_model(use_gpu: bool, random_state: int = RANDOM_SEED) -> XGBRegressor:
    """Create the fixed XGBoost regressor used throughout the study."""
    return XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
    )


def gpu_is_available(disable_gpu: bool) -> bool:
    if disable_gpu:
        return False
    try:
        probe_x = pd.DataFrame(np.zeros((24, 2)), columns=["a", "b"])
        probe_y = np.arange(24, dtype=float)
        create_xgb_model(True).fit(probe_x, probe_y)
        return True
    except Exception:
        return False


def sample_weights(target: pd.Series) -> pd.Series:
    """Return the study-specific inverse-distance training weights."""
    return 1 / (np.abs(target - target.mean()) + 0.1)


def save_parity_plot(actual: pd.Series, predicted: np.ndarray, path: Path) -> None:
    plt.close("all")
    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.4)
    bounds = [actual.min(), actual.max()]
    plt.plot(bounds, bounds, color="red")
    plt.xlabel("Actual log10(E)")
    plt.ylabel("Predicted log10(E)")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def run_analysis(
    features: pd.DataFrame,
    descriptor_columns: list[str],
    processed_dir: Path,
    output_dir: Path,
    figure_dir: Path,
    use_gpu: bool,
) -> dict[str, object]:
    """Train, evaluate and interpret the fixed XGBoost workflow."""
    target = features["logE"]
    predictors_raw = features.drop(columns=["logE", "E", "K", "G"], errors="ignore")
    leakage_columns = {"logE", "E", "K", "G"}
    prohibited = {
        "density",
        "density_atomic",
        "homogeneous_poisson",
        "universal_anisotropy",
        "volume",
        "nsites",
    }
    if leakage_columns.intersection(predictors_raw.columns):
        raise RuntimeError("Target-derived columns remain in the predictor matrix.")
    if prohibited.intersection(predictors_raw.columns):
        raise RuntimeError("A non-composition Materials Project field was detected.")
    if set(predictors_raw.columns) != set(descriptor_columns):
        raise RuntimeError("Unexpected predictor columns were detected.")

    raw_train, raw_test, y_train, y_test = train_test_split(
        predictors_raw, target, test_size=0.2, random_state=RANDOM_SEED
    )
    x_train, x_test = prepare_features(raw_train, raw_test)

    cv_scores: list[float] = []
    folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, validation_index in folds.split(raw_train):
        fold_raw_train = raw_train.iloc[train_index]
        fold_raw_validation = raw_train.iloc[validation_index]
        fold_x_train, fold_x_validation = prepare_features(
            fold_raw_train, fold_raw_validation
        )
        fold_y_train = y_train.iloc[train_index]
        fold_y_validation = y_train.iloc[validation_index]
        model = create_xgb_model(use_gpu)
        model.fit(
            fold_x_train,
            fold_y_train,
            sample_weight=sample_weights(fold_y_train),
        )
        cv_scores.append(r2_score(fold_y_validation, model.predict(fold_x_validation)))

    model = create_xgb_model(use_gpu)
    weights = sample_weights(y_train)
    model.fit(x_train, y_train, sample_weight=weights)
    predictions = model.predict(x_test)
    random_metrics = {
        "r2": r2_score(y_test, predictions),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": mean_absolute_error(y_test, predictions),
    }

    pd.DataFrame(
        {
            "Metric": [
                "5-fold CV R² mean",
                "5-fold CV R² std",
                "Test R²",
                "RMSE (log10(E))",
                "MAE (log10(E))",
            ],
            "Value": [
                np.mean(cv_scores),
                np.std(cv_scores),
                random_metrics["r2"],
                random_metrics["rmse"],
                random_metrics["mae"],
            ],
        }
    ).to_csv(output_dir / "table3_model_performance.csv", index=False)
    pd.DataFrame(
        {
            "Actual_log10E": y_test.to_numpy(),
            "Predicted_log10E": predictions,
            "row_index": y_test.index,
        }
    ).to_csv(output_dir / "model_predictions_test_set.csv", index=False)
    save_parity_plot(y_test, predictions, figure_dir / "parity_final.png")

    gain_importance = pd.DataFrame(
        {"Feature": x_train.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    gain_importance.to_csv(output_dir / "top_feature_importance.csv", index=False)
    top_twenty = gain_importance.head(20).sort_values("Importance")
    plt.figure(figsize=(8, 6))
    plt.barh(top_twenty["Feature"], top_twenty["Importance"])
    plt.xlabel("XGBoost feature importance")
    plt.tight_layout()
    plt.savefig(
        figure_dir / "feature_importance_final.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    shap_sample = x_test.sample(min(500, len(x_test)), random_state=RANDOM_SEED)
    shap_values = shap.TreeExplainer(model).shap_values(shap_sample)
    shap.summary_plot(
        shap_values,
        shap_sample,
        feature_names=shap_sample.columns,
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    shap_importance = pd.DataFrame(
        {
            "Feature": shap_sample.columns,
            "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("Mean_abs_SHAP", ascending=False)
    shap_importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    shap.summary_plot(
        shap_values,
        shap_sample,
        feature_names=shap_sample.columns,
        plot_type="bar",
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figure_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    ablation_records: list[dict[str, object]] = []
    ablation_records.append(
        {"Model": "Full features", "Number of features": x_train.shape[1], "R2": random_metrics["r2"]}
    )
    top_columns = gain_importance.head(30)["Feature"].tolist()
    top_model = create_xgb_model(use_gpu)
    top_model.fit(x_train[top_columns], y_train, sample_weight=weights)
    ablation_records.append(
        {
            "Model": "Top 30 features",
            "Number of features": 30,
            "R2": r2_score(y_test, top_model.predict(x_test[top_columns])),
        }
    )
    gain_importance.head(30).to_csv(
        output_dir / "top30_features_used_in_ablation.csv", index=False
    )
    generator = np.random.default_rng(RANDOM_SEED)
    random_columns = generator.choice(x_train.columns.to_numpy(), 30, replace=False)
    random_model = create_xgb_model(use_gpu)
    random_model.fit(x_train[random_columns], y_train, sample_weight=weights)
    ablation_records.append(
        {
            "Model": "Random 30 features",
            "Number of features": 30,
            "R2": r2_score(y_test, random_model.predict(x_test[random_columns])),
        }
    )
    pd.DataFrame({"Random_30_Features": random_columns}).to_csv(
        output_dir / "random30_features_used_in_ablation.csv", index=False
    )
    pd.DataFrame(ablation_records).to_csv(
        output_dir / "table4_ablation_study.csv", index=False
    )

    robustness_records: list[dict[str, float | int]] = []
    for seed in ROBUSTNESS_SEEDS:
        split_raw_train, split_raw_test, split_y_train, split_y_test = train_test_split(
            predictors_raw, target, test_size=0.2, random_state=seed
        )
        split_x_train, split_x_test = prepare_features(split_raw_train, split_raw_test)
        split_model = create_xgb_model(use_gpu, random_state=seed)
        split_model.fit(
            split_x_train,
            split_y_train,
            sample_weight=sample_weights(split_y_train),
        )
        robustness_records.append(
            {
                "Random seed": seed,
                "Test R2": r2_score(split_y_test, split_model.predict(split_x_test)),
            }
        )
    robustness = pd.DataFrame(robustness_records)
    robustness.to_csv(output_dir / "robustness_scores_by_seed.csv", index=False)
    pd.DataFrame(
        {
            "Metric": ["Mean R2", "Std R2"],
            "Value": [
                robustness["Test R2"].mean(),
                robustness["Test R2"].std(ddof=0),
            ],
        }
    ).to_csv(output_dir / "table5_robustness_summary.csv", index=False)
    plt.figure(figsize=(7, 5))
    plt.plot(
        robustness["Random seed"].astype(str), robustness["Test R2"], marker="o"
    )
    plt.axhline(robustness["Test R2"].mean(), linestyle="--", label="Mean R²")
    plt.xlabel("Random seed")
    plt.ylabel("Test R²")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "robustness_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    formula_groups = pd.read_csv(processed_dir / "composition_groups.csv")[
        "reduced_formula"
    ]
    grouped_split = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_SEED
    )
    grouped_train_index, grouped_test_index = next(
        grouped_split.split(predictors_raw, target, groups=formula_groups)
    )
    if set(formula_groups.iloc[grouped_train_index]).intersection(
        formula_groups.iloc[grouped_test_index]
    ):
        raise RuntimeError("Reduced-formula leakage was detected in the grouped split.")
    grouped_raw_train = predictors_raw.iloc[grouped_train_index]
    grouped_raw_test = predictors_raw.iloc[grouped_test_index]
    grouped_y_train = target.iloc[grouped_train_index]
    grouped_y_test = target.iloc[grouped_test_index]
    grouped_x_train, grouped_x_test = prepare_features(
        grouped_raw_train, grouped_raw_test
    )
    grouped_model = create_xgb_model(use_gpu)
    grouped_model.fit(
        grouped_x_train,
        grouped_y_train,
        sample_weight=sample_weights(grouped_y_train),
    )
    grouped_predictions = grouped_model.predict(grouped_x_test)
    grouped_metrics = {
        "r2": r2_score(grouped_y_test, grouped_predictions),
        "rmse": float(
            np.sqrt(mean_squared_error(grouped_y_test, grouped_predictions))
        ),
        "mae": mean_absolute_error(grouped_y_test, grouped_predictions),
    }
    pd.DataFrame(
        {
            "Metric": [
                "Grouped test R²",
                "Grouped RMSE (log10(E))",
                "Grouped MAE (log10(E))",
                "Train samples",
                "Test samples",
                "Train formulas",
                "Test formulas",
            ],
            "Value": [
                grouped_metrics["r2"],
                grouped_metrics["rmse"],
                grouped_metrics["mae"],
                len(grouped_train_index),
                len(grouped_test_index),
                formula_groups.iloc[grouped_train_index].nunique(),
                formula_groups.iloc[grouped_test_index].nunique(),
            ],
        }
    ).to_csv(output_dir / "composition_grouped_evaluation.csv", index=False)

    return {
        "cv_scores": cv_scores,
        "random_metrics": random_metrics,
        "grouped_metrics": grouped_metrics,
        "random_feature_count": x_train.shape[1],
        "grouped_feature_count": grouped_x_train.shape[1],
    }


def export_summary_tables(
    counts: dict[str, int],
    features: pd.DataFrame,
    analysis: dict[str, object],
    output_dir: Path,
) -> None:
    pd.DataFrame(
        {
            "Stage": [
                "Raw dataset",
                "After removing failed calculations",
                "After physical consistency filtering",
                "Final machine learning dataset after feature generation",
            ],
            "Number of samples": [
                counts["raw"],
                counts["successful"],
                counts["target_filtered"],
                len(features),
            ],
        }
    ).to_csv(output_dir / "table1_dataset_cleaning_summary.csv", index=False)

    pd.DataFrame(
        {
            "Stage": [
                "Numeric composition-descriptor dataset including target",
                "Composition descriptors after target removal",
                "Final input features after correlation filtering",
            ],
            "Number of features": [
                features.shape[1],
                features.drop(columns=["logE", "E", "K", "G"], errors="ignore").shape[1],
                analysis["random_feature_count"],
            ],
        }
    ).to_csv(output_dir / "table2_feature_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    run_directory = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = run_directory / "processed"
    output_dir = run_directory / "outputs"
    figure_dir = run_directory / "figures"
    for directory in (processed_dir, output_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=False)

    clean_frame, counts = clean_dataset(args.raw_data, processed_dir)
    features, descriptor_columns = generate_descriptors(clean_frame, processed_dir)
    use_gpu = gpu_is_available(args.cpu_only)
    analysis = run_analysis(
        features,
        descriptor_columns,
        processed_dir,
        output_dir,
        figure_dir,
        use_gpu,
    )
    export_summary_tables(counts, features, analysis, output_dir)

    provenance = {
        "raw_data": str(args.raw_data.resolve()),
        "run_directory": str(run_directory.resolve()),
        "target": "log10(E in GPa)",
        "random_seed": RANDOM_SEED,
        "robustness_seeds": list(ROBUSTNESS_SEEDS),
        "gpu_used": use_gpu,
        "descriptor_count": len(descriptor_columns),
        "sample_count": len(features),
    }
    with (run_directory / "run_provenance.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(provenance, handle, indent=2)

    print(f"Analysis completed: {run_directory}")
    print("Random held-out metrics:", analysis["random_metrics"])
    print("Formula-grouped metrics:", analysis["grouped_metrics"])


if __name__ == "__main__":
    main()
