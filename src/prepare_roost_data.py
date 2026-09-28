"""Prepare the exact train, validation and test files used for Roost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def export_partition(
    label: str,
    training_indices: np.ndarray,
    test_indices: np.ndarray,
    features: pd.DataFrame,
    groups: pd.DataFrame,
    output_root: Path,
) -> dict[str, object]:
    partition_dir = output_root / label
    partition_dir.mkdir(parents=True, exist_ok=False)
    row_ids = np.arange(len(features))
    data = pd.DataFrame(
        {
            "material_id": [f"hana2-{index}" for index in row_ids],
            "composition": groups["formula_pretty"].astype(str).to_numpy(),
            "logE": features["logE"].to_numpy(),
            "row_index": row_ids,
            "reduced_formula": groups["reduced_formula"].astype(str).to_numpy(),
        }
    )

    nominal_train = data.iloc[training_indices].reset_index(drop=True)
    if label == "grouped":
        validation_split = GroupShuffleSplit(
            n_splits=1, test_size=0.1, random_state=43
        )
        fit_indices, validation_indices = next(
            validation_split.split(
                nominal_train,
                groups=nominal_train["reduced_formula"],
            )
        )
        if set(nominal_train.iloc[fit_indices]["reduced_formula"]).intersection(
            nominal_train.iloc[validation_indices]["reduced_formula"]
        ):
            raise RuntimeError("Formula overlap was detected in grouped validation.")
    else:
        fit_indices, validation_indices = train_test_split(
            np.arange(len(nominal_train)), test_size=0.1, random_state=43
        )

    roost_columns = ["material_id", "composition", "logE"]
    nominal_train.iloc[fit_indices][roost_columns].to_csv(
        partition_dir / "roost_fit.csv", index=False
    )
    nominal_train.iloc[validation_indices][roost_columns].to_csv(
        partition_dir / "roost_val.csv", index=False
    )
    data.iloc[test_indices][roost_columns].to_csv(
        partition_dir / "roost_test.csv", index=False
    )
    data.iloc[test_indices].to_csv(
        partition_dir / "test_row_mapping.csv", index=False
    )

    formula_overlap = len(
        set(data.iloc[training_indices]["reduced_formula"]).intersection(
            data.iloc[test_indices]["reduced_formula"]
        )
    )
    return {
        "split": label,
        "n_nominal_train": len(training_indices),
        "n_fit": len(fit_indices),
        "n_validation": len(validation_indices),
        "n_test": len(test_indices),
        "formula_overlap": formula_overlap,
        "validation_seed": 43,
    }


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.processed_dir / "mp_features.csv")
    groups = pd.read_csv(args.processed_dir / "composition_groups.csv")
    if len(features) != len(groups):
        raise ValueError("Feature and formula metadata rows are not aligned.")
    if not {"formula_pretty", "reduced_formula"}.issubset(groups.columns):
        raise ValueError("composition_groups.csv has an unexpected schema.")

    row_ids = np.arange(len(features))
    random_train, random_test = train_test_split(
        row_ids, test_size=0.2, random_state=42
    )
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    grouped_train, grouped_test = next(
        splitter.split(features, features["logE"], groups["reduced_formula"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = [
        export_partition(
            "random",
            random_train,
            random_test,
            features,
            groups,
            args.output_dir,
        ),
        export_partition(
            "grouped",
            grouped_train,
            grouped_test,
            features,
            groups,
            args.output_dir,
        ),
    ]
    if manifest[1]["formula_overlap"] != 0:
        raise RuntimeError("Formula overlap was detected in the grouped test split.")

    with (args.output_dir / "partition_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "target": "log10(E in GPa)",
                "same_test_splits_as_xgboost": True,
                "partitions": manifest,
            },
            handle,
            indent=2,
        )
    print(f"Roost input files written to {args.output_dir}")


if __name__ == "__main__":
    main()
