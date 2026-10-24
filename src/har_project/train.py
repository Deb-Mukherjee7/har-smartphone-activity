"""Compare models with/without PCA using subject-wise cross-validation and held-out test data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from har_project.data import download_dataset, load_dataset
from har_project.features import GroupedStackingClassifier, build_classifier

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


def _transform_fit(X: pd.DataFrame, variance_threshold: float | None):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    pca = None
    if variance_threshold is not None:
        pca = PCA(n_components=variance_threshold, svd_solver="full")
        transformed = pca.fit_transform(scaled)
    else:
        transformed = scaled
    return scaler, pca, transformed


def _transform(scaler, pca, X: pd.DataFrame):
    scaled = scaler.transform(X)
    return pca.transform(scaled) if pca is not None else scaled


def _new_model(name: str, seed: int, stack_folds: int):
    if name == "stacked_ensemble":
        return GroupedStackingClassifier(random_seed=seed, n_splits=stack_folds)
    return build_classifier(name, random_seed=seed)


def _fit(model, X, y, groups):
    if isinstance(model, GroupedStackingClassifier):
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def _score(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _representation_specs(config: dict):
    specs = [("no_pca", None)]
    for threshold in config["models"]["pca_variance_thresholds"]:
        value = float(threshold)
        label = (f"{value * 100:g}").replace(".", "_")
        specs.append((f"pca_{label}", value))
    return specs


def run_experiment(config: dict, dataset_dir: Path | None = None) -> pd.DataFrame:
    dataset_dir = Path(dataset_dir or ROOT / config["data"]["dataset_dir"])
    if not (dataset_dir / "features.txt").is_file():
        dataset_dir = download_dataset(
            ROOT / config["data"]["raw_dir"], ROOT / config["data"]["zip_path"]
        )
    data = load_dataset(dataset_dir)
    report_dir = ROOT / config["outputs"]["report_dir"]
    model_dir = ROOT / config["outputs"]["model_dir"]
    (report_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["project"]["random_seed"])
    n_splits = int(config["validation"]["folds"])
    stack_folds = min(n_splits, 5)
    candidates = list(config["models"]["candidates"])
    subjects = data.subject_train.to_numpy()
    y_train = data.y_train.to_numpy()
    y_test = data.y_test.to_numpy()
    fold_splitter = GroupKFold(n_splits=n_splits)
    rows = []

    # Fit a separate representation for every outer fold so scaling and PCA never see
    # that fold's validation subjects.
    for representation, threshold in _representation_specs(config):
        print(f"\n=== Representation: {representation} ===")
        fold_records = {name: [] for name in candidates}
        fold_train_records = {name: [] for name in candidates}
        fold_fit_seconds = {name: [] for name in candidates}
        for fold_number, (fit_idx, validation_idx) in enumerate(
            fold_splitter.split(data.X_train, y_train, groups=subjects), start=1
        ):
            X_fit_raw = data.X_train.iloc[fit_idx]
            X_validation_raw = data.X_train.iloc[validation_idx]
            scaler, pca, X_fit = _transform_fit(X_fit_raw, threshold)
            X_validation = _transform(scaler, pca, X_validation_raw)
            y_fit, y_validation = y_train[fit_idx], y_train[validation_idx]
            groups_fit = subjects[fit_idx]

            for name in candidates:
                model = _new_model(name, seed, stack_folds)
                started = time.perf_counter()
                _fit(model, X_fit, y_fit, groups_fit)
                fit_seconds = time.perf_counter() - started
                train_predicted = model.predict(X_fit)
                fold_train_records[name].append(_score(y_fit, train_predicted))
                predicted = model.predict(X_validation)
                fold_records[name].append(_score(y_validation, predicted))
                fold_fit_seconds[name].append(fit_seconds)
            print(f"  completed grouped fold {fold_number}/{n_splits}")

        # Fit every requested model on the complete official training subjects, then
        # evaluate once on the official held-out subjects.
        scaler, pca, X_fit = _transform_fit(data.X_train, threshold)
        X_test = _transform(scaler, pca, data.X_test)
        retained = int(pca.n_components_) if pca is not None else int(data.X_train.shape[1])
        variance_kept = (
            float(pca.explained_variance_ratio_.sum()) if pca is not None else 1.0
        )

        for name in candidates:
            model = _new_model(name, seed, stack_folds)
            started = time.perf_counter()
            _fit(model, X_fit, y_train, subjects)
            final_fit_seconds = time.perf_counter() - started

            # Benchmark repeated small batches to reduce single-call timer noise.
            batch_size = min(256, len(X_test))
            prediction_batch = X_test[:batch_size]
            repetitions = 8
            started = time.perf_counter()
            for _ in range(repetitions):
                model.predict(prediction_batch)
            latency_ms = (
                (time.perf_counter() - started) / (repetitions * batch_size) * 1000
            )

            test_prediction = model.predict(X_test)
            scores = _score(y_test, test_prediction)
            artifact_path = model_dir / representation / f"{name}.joblib"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            bundle = {
                "model": model,
                "scaler": scaler,
                "pca": pca,
                "model_name": name,
                "representation": representation,
                "feature_names": list(data.X_train.columns),
                "activities": list(model.classes_),
                "pca_components": retained,
                "retained_variance": variance_kept,
            }
            joblib.dump(bundle, artifact_path, compress=3)
            model_size_mb = artifact_path.stat().st_size / (1024 * 1024)

            cv_f1 = [item["macro_f1"] for item in fold_records[name]]
            cv_accuracy = [item["accuracy"] for item in fold_records[name]]
            train_f1 = [item["macro_f1"] for item in fold_train_records[name]]
            train_accuracy = [item["accuracy"] for item in fold_train_records[name]]
            f1_gaps = [train - validation for train, validation in zip(train_f1, cv_f1)]
            row = {
                "representation": representation,
                "model": name,
                **scores,
                "cv_train_macro_f1_mean": float(np.mean(train_f1)),
                "cv_train_accuracy_mean": float(np.mean(train_accuracy)),
                "cv_macro_f1_mean": float(np.mean(cv_f1)),
                "cv_macro_f1_std": float(np.std(cv_f1, ddof=1)),
                "cv_macro_f1_gap_mean": float(np.mean(f1_gaps)),
                "cv_accuracy_mean": float(np.mean(cv_accuracy)),
                "grouped_cv_mean_fit_seconds": float(np.mean(fold_fit_seconds[name])),
                "final_train_seconds": float(final_fit_seconds),
                "inference_latency_ms_per_sample": float(latency_ms),
                "model_size_mb": float(model_size_mb),
                "pca_components": retained,
                "retained_variance": variance_kept,
                "model_path": str(artifact_path.relative_to(ROOT)),
            }
            rows.append(row)

            matrix = confusion_matrix(y_test, test_prediction, labels=ACTIVITY_ORDER)
            plt.figure(figsize=(8.8, 7.0))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=ACTIVITY_ORDER,
                yticklabels=ACTIVITY_ORDER,
            )
            plt.title(f"Official test subjects: {representation} / {name}")
            plt.xlabel("Predicted activity")
            plt.ylabel("True activity")
            plt.xticks(rotation=35, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(
                report_dir / "confusion_matrices" / f"{representation}_{name}.png", dpi=130
            )
            plt.close()
            print(
                f"  {name}: test macro-F1={scores['macro_f1']:.4f}, "
                f"CV macro-F1={np.mean(cv_f1):.4f}, "
                f"train-validation gap={np.mean(f1_gaps):.4f}, "
                f"accuracy={scores['accuracy']:.4f}, components={retained}"
            )

        if pca is not None:
            pd.DataFrame(
                {
                    "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                    "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
                }
            ).to_csv(report_dir / f"{representation}_explained_variance.csv", index=False)

    results = pd.DataFrame(rows).sort_values(
        ["representation", "macro_f1"], ascending=[True, False]
    )
    results.to_csv(report_dir / "model_comparison.csv", index=False)
    results.sort_values("macro_f1", ascending=False).drop_duplicates("representation").sort_values(
        "representation"
    ).to_csv(report_dir / "representation_summary.csv", index=False)

    # Two-dimensional PCA visualization is fit on training subjects only. It is separate
    # from each classifier's variance-threshold PCA and is used only by the demo.
    demo_scaler = StandardScaler().fit(data.X_train)
    train_scaled = demo_scaler.transform(data.X_train)
    test_scaled = demo_scaler.transform(data.X_test)
    visual_pca = PCA(n_components=2, svd_solver="full", random_state=seed)
    train_xy = visual_pca.fit_transform(train_scaled)
    test_xy = visual_pca.transform(test_scaled)
    pd.DataFrame(
        {
            "pc1": train_xy[:, 0],
            "pc2": train_xy[:, 1],
            "activity": data.y_train.to_numpy(),
            "subject": data.subject_train.to_numpy(),
        }
    ).to_csv(report_dir / "training_pca_2d.csv", index=False)
    pd.DataFrame(
        {
            "pc1": test_xy[:, 0],
            "pc2": test_xy[:, 1],
            "activity": data.y_test.to_numpy(),
            "subject": data.subject_test.to_numpy(),
        }
    ).to_csv(report_dir / "test_pca_2d.csv", index=False)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pd.DataFrame({"PC1": train_xy[:, 0], "PC2": train_xy[:, 1], "Activity": data.y_train}),
        x="PC1",
        y="PC2",
        hue="Activity",
        alpha=0.55,
        s=20,
    )
    plt.title("Training samples projected onto first two PCA components")
    plt.tight_layout()
    plt.savefig(report_dir / "training_pca_2d.png", dpi=150)
    plt.close()

    joblib.dump(
        {
            "scaler": demo_scaler,
            "pca": visual_pca,
            "feature_names": list(data.X_train.columns),
            "activities": ACTIVITY_ORDER,
            "train_embedding": train_xy,
            "train_labels": data.y_train.to_numpy(),
            "test_embedding": test_xy,
            "test_labels": data.y_test.to_numpy(),
        },
        model_dir / "demo_assets.joblib",
        compress=3,
    )

    metadata = {
        "project": config["project"]["name"],
        "dataset_dir": str(dataset_dir),
        "train_rows": len(data.y_train),
        "test_rows": len(data.y_test),
        "train_subjects": sorted(int(x) for x in data.subject_train.unique()),
        "test_subjects": sorted(int(x) for x in data.subject_test.unique()),
        "pca_visualization_explained_variance": visual_pca.explained_variance_ratio_.tolist(),
        "models_compared": candidates,
        "representations_compared": [name for name, _ in _representation_specs(config)],
    }
    with (report_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"\nSaved comparison: {report_dir / 'model_comparison.csv'}")
    print(f"Saved interactive demo model assets: {model_dir / 'demo_assets.joblib'}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare configured classifiers with and without PCA using subject-wise cross-validation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    run_experiment(config, args.dataset_dir)


if __name__ == "__main__":
    main()
