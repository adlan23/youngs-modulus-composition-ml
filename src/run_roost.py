"""Run and evaluate the pinned Aviary/Roost implementation.

Run ``prepare_roost_data.py`` first. Aviary must be installed in a separate
environment and checked out at the commit recorded in ``AVIARY_COMMIT``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


AVIARY_COMMIT = "3f4db21c2b807ebe993f02d8ba13aafbdd7e07a1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aviary-dir",
        type=Path,
        required=True,
        help="Aviary repository checked out at the pinned commit.",
    )
    parser.add_argument(
        "--roost-data-dir",
        type=Path,
        required=True,
        help="Output directory produced by prepare_roost_data.py.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable from the environment containing Aviary.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def verify_aviary(aviary_dir: Path) -> Path:
    script = aviary_dir / "examples" / "roost-example.py"
    if not script.is_file():
        raise FileNotFoundError(f"Roost example script not found: {script}")
    commit = subprocess.check_output(
        ["git", "-C", str(aviary_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != AVIARY_COMMIT:
        raise RuntimeError(
            f"Aviary commit {commit} does not match the reference commit "
            f"{AVIARY_COMMIT}."
        )
    return script


def locate_prediction_file(aviary_dir: Path, model_name: str) -> Path:
    candidates = [
        path
        for path in aviary_dir.rglob("*.csv")
        if model_name in path.name and path.stat().st_size < 20_000_000
    ]
    valid: list[Path] = []
    for path in candidates:
        try:
            columns = pd.read_csv(path, nrows=2).columns
        except Exception:
            continue
        if "material_id" in columns and "logE_preds_n0" in columns:
            valid.append(path)
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected one prediction file for {model_name}; found {valid}."
        )
    return valid[0]


def run_partition(
    label: str,
    data_root: Path,
    aviary_dir: Path,
    script: Path,
    python_executable: Path,
    epochs: int,
) -> dict[str, object]:
    partition = data_root / label
    required = [
        partition / "roost_fit.csv",
        partition / "roost_val.csv",
        partition / "roost_test.csv",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required Roost input not found: {path}")

    model_name = f"youngs_modulus_{label}"
    command = [
        str(python_executable),
        str(script),
        "--train",
        "--evaluate",
        "--data-path",
        str(partition / "roost_fit.csv"),
        "--val-path",
        str(partition / "roost_val.csv"),
        "--test-path",
        str(partition / "roost_test.csv"),
        "--targets",
        "logE",
        "--tasks",
        "regression",
        "--losses",
        "L1",
        "--model-name",
        model_name,
        "--epochs",
        str(epochs),
        "--data-seed",
        "42",
        "--workers",
        "0",
    ]
    with (partition / "roost_command.json").open("w", encoding="utf-8") as handle:
        json.dump(command, handle, indent=2)
    with (partition / "roost_training.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=aviary_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

    predictions_path = locate_prediction_file(aviary_dir, model_name)
    expected = pd.read_csv(partition / "roost_test.csv")
    predicted = pd.read_csv(predictions_path)
    if not expected["material_id"].is_unique or not predicted["material_id"].is_unique:
        raise RuntimeError("Roost material identifiers are not unique.")
    paired = expected[["material_id", "logE"]].merge(
        predicted[["material_id", "logE_preds_n0"]],
        on="material_id",
        validate="one_to_one",
    )
    if len(paired) != len(expected):
        raise RuntimeError("Roost predictions do not match every held-out record.")

    observed = paired["logE"].to_numpy()
    estimates = paired["logE_preds_n0"].to_numpy()
    paired.to_csv(partition / "roost_predictions.csv", index=False)
    return {
        "Split": label,
        "Model": "Roost",
        "Test_samples": len(paired),
        "R2": r2_score(observed, estimates),
        "RMSE_log10E": float(np.sqrt(mean_squared_error(observed, estimates))),
        "MAE_log10E": mean_absolute_error(observed, estimates),
        "Prediction_file": str(predictions_path),
    }


def main() -> None:
    args = parse_args()
    script = verify_aviary(args.aviary_dir)
    results = [
        run_partition(
            label,
            args.roost_data_dir,
            args.aviary_dir,
            script,
            args.python,
            args.epochs,
        )
        for label in ("random", "grouped")
    ]
    scores = pd.DataFrame(results)
    scores.to_csv(args.roost_data_dir / "roost_scores.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
