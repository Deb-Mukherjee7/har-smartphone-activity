"""Report model performance separately for each held-out test participant."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from har_project.data import download_dataset, load_dataset

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
ACTIVITY_ORDER = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def run_subject_report(config: dict, dataset_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = Path(dataset_dir or ROOT / config["data"]["dataset_dir"])
    if not (dataset_dir / "features.txt").is_file():
        dataset_dir = download_dataset(
            ROOT / config["data"]["raw_dir"], ROOT / config["data"]["zip_path"]
        )

    data = load_dataset(dataset_dir)
    model_dir = ROOT / config["outputs"]["model_dir"]
    report_dir = ROOT / config["outputs"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    candidates = set(config["models"]["candidates"])
    model_paths = sorted(
        path for path in model_dir.glob("*/*.joblib") if path.stem in candidates
    )
    if not model_paths:
        raise FileNotFoundError(
            f"No trained model files found in {model_dir}. Run training first."
        )

    y_test = data.y_test.to_numpy()
    subject_ids = data.subject_test.to_numpy()
    rows = []

    for model_path in model_paths:
        bundle = joblib.load(model_path)
        X_raw = data.X_test.loc[:, bundle["feature_names"]]
        scaled = bundle["scaler"].transform(X_raw)
        X_model = bundle["pca"].transform(scaled) if bundle["pca"] is not None else scaled
        predictions = bundle["model"].predict(X_model)

        for subject_id in np.unique(subject_ids):
            mask = subject_ids == subject_id
            rows.append(
                {
                    "representation": bundle["representation"],
                    "model": bundle["model_name"],
                    "subject_id": int(subject_id),
                    "sample_count": int(mask.sum()),
                    "accuracy": float(accuracy_score(y_test[mask], predictions[mask])),
                    "macro_f1": float(
                        f1_score(
                            y_test[mask],
                            predictions[mask],
                            labels=ACTIVITY_ORDER,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(y_test[mask], predictions[mask])
                    ),
                }
            )

    per_subject = pd.DataFrame(rows).sort_values(
        ["representation", "model", "subject_id"]
    )
    per_subject.to_csv(report_dir / "subject_performance.csv", index=False)

    summary = (
        per_subject.groupby(["representation", "model"], as_index=False)
        .agg(
            subjects_tested=("subject_id", "nunique"),
            mean_subject_accuracy=("accuracy", "mean"),
            std_subject_accuracy=("accuracy", "std"),
            lowest_subject_accuracy=("accuracy", "min"),
            mean_subject_macro_f1=("macro_f1", "mean"),
            std_subject_macro_f1=("macro_f1", "std"),
            lowest_subject_macro_f1=("macro_f1", "min"),
        )
        .sort_values(["representation", "mean_subject_macro_f1"], ascending=[True, False])
    )
    summary.to_csv(report_dir / "subject_performance_summary.csv", index=False)

    print(f"Saved participant-level scores: {report_dir / 'subject_performance.csv'}")
    print(f"Saved participant summary: {report_dir / 'subject_performance_summary.csv'}")
    print("These are descriptive scores on the official held-out test subjects.")
    return per_subject, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report activity-recognition scores for each official test participant."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_subject_report(config, args.dataset_dir)


if __name__ == "__main__":
    main()
