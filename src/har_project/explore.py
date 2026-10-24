"""Create descriptive dataset reports and PCA variance plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from har_project.data import download_dataset, load_dataset

ROOT = Path(__file__).resolve().parents[2]


def explore(dataset_dir: Path, output_dir: Path) -> None:
    data = load_dataset(dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data.y_train.value_counts().rename_axis("activity").rename("count").to_csv(
        output_dir / "class_balance.csv"
    )
    data.X_train.isna().sum().rename("missing_values").to_csv(
        output_dir / "missing_values_by_feature.csv"
    )

    plt.figure(figsize=(9, 5))
    order = data.y_train.value_counts().index
    sns.countplot(x=data.y_train, order=order, color="#4677a8")
    plt.title("Training set class balance")
    plt.xlabel("Activity")
    plt.ylabel("Window count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=150)
    plt.close()

    scaler = StandardScaler().fit(data.X_train)
    standardized = scaler.transform(data.X_train)
    pca = PCA(svd_solver="full").fit(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    )
    variance.to_csv(output_dir / "pca_explained_variance.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.plot(variance["component"], cumulative, color="#245b83")
    for threshold in (0.95, 0.99, 0.995):
        components = int(np.searchsorted(cumulative, threshold) + 1)
        plt.axhline(threshold, linestyle="--", linewidth=0.8, alpha=0.7)
        plt.axvline(components, linestyle=":", linewidth=0.8, alpha=0.7)
        plt.text(components, threshold, f" {threshold:.1%}: {components} PCs", fontsize=8)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA explained variance (scaler fitted on training data)")
    plt.ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "pca_explained_variance.png", dpi=150)
    plt.close()

    # Keep the heatmap readable: show the 30 features with highest training variance.
    selected = data.X_train.var().nlargest(30).index
    corr = data.X_train.loc[:, selected].corr()
    corr.to_csv(output_dir / "top30_feature_correlations.csv")
    plt.figure(figsize=(14, 11))
    sns.heatmap(corr, cmap="vlag", center=0, xticklabels=True, yticklabels=True)
    plt.title("Correlation among 30 highest-variance training features")
    plt.tight_layout()
    plt.savefig(output_dir / "top30_feature_correlations.png", dpi=150)
    plt.close()
    print(f"Exploration files written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate UCI HAR exploratory data reports.")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/raw/UCI HAR Dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/eda")
    args = parser.parse_args()
    if not (args.dataset_dir / "features.txt").is_file():
        args.dataset_dir = download_dataset(
            args.dataset_dir.parent, args.dataset_dir.parent / "uci_har.zip"
        )
    explore(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
