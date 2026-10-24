"""Apply a saved model to a CSV of UCI-compatible feature rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def predict_file(model_path: Path, input_path: Path, output_path: Path) -> None:
    """Predict activity labels for a headered CSV containing the expected feature columns."""

    bundle = joblib.load(model_path)
    expected = bundle["feature_names"]
    frame = pd.read_csv(input_path)
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Input CSV is missing {len(missing)} required feature columns. "
            f"Examples: {missing[:5]}. Use the 561 UCI feature columns in their original names."
        )
    X = frame.loc[:, expected]
    if X.isna().any().any():
        raise ValueError("Input feature CSV contains missing values.")
    if "scaler" in bundle:
        transformed = bundle["scaler"].transform(X)
        if bundle.get("pca") is not None:
            transformed = bundle["pca"].transform(transformed)
    else:  # Backward-compatible handling for an older direct-model bundle.
        transformed = X
    prediction = bundle["model"].predict(transformed)
    result = pd.DataFrame({"predicted_activity": prediction})
    if hasattr(bundle["model"], "predict_proba"):
        probabilities = bundle["model"].predict_proba(transformed)
        classes = bundle["model"].classes_
        for column, class_name in enumerate(classes):
            safe_name = str(class_name).lower().replace(" ", "_")
            result[f"probability_{safe_name}"] = probabilities[:, column]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict activities from a feature CSV.")
    parser.add_argument("input_csv", type=Path, help="Headered CSV with UCI feature names.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "predictions.csv")
    args = parser.parse_args()
    predict_file(args.model, args.input_csv, args.output)


if __name__ == "__main__":
    main()
