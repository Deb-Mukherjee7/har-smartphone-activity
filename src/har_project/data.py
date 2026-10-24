"""Download and load the original UCI HAR Dataset files."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_ZIP_PATH = DEFAULT_RAW_DIR / "uci_har.zip"
EXPECTED_FEATURE_COUNT = 561
EXPECTED_ACTIVITIES = 6


@dataclass
class HARData:
    """The official train/test feature matrices, labels, and subject IDs."""

    X_train: pd.DataFrame
    y_train: pd.Series
    subject_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    subject_test: pd.Series


def download_dataset(raw_dir: Path = DEFAULT_RAW_DIR, zip_path: Path = DEFAULT_ZIP_PATH) -> Path:
    """Download and safely extract the official archive; return its dataset directory."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = raw_dir / "UCI HAR Dataset"
    if dataset_dir.exists() and (dataset_dir / "features.txt").exists():
        print(f"Dataset already exists: {dataset_dir}")
        return dataset_dir

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading UCI dataset from {UCI_ZIP_URL}")
        request = urllib.request.Request(
            UCI_ZIP_URL, headers={"User-Agent": "har-smartphone-activity/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response, zip_path.open("wb") as out:
            shutil.copyfileobj(response, out)

    with zipfile.ZipFile(zip_path) as archive:
        root = Path(raw_dir).resolve()
        for member in archive.infolist():
            target = (Path(raw_dir) / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
        archive.extractall(raw_dir)

    if not (dataset_dir / "features.txt").exists():
        # Some mirrors add a top-level folder; locate the canonical directory.
        candidates = list(Path(raw_dir).rglob("UCI HAR Dataset/features.txt"))
        if not candidates:
            raise FileNotFoundError("Could not find 'UCI HAR Dataset/features.txt' after extraction.")
        dataset_dir = candidates[0].parent
    print(f"Dataset extracted to: {dataset_dir}")
    return dataset_dir


def _read_vector(path: Path) -> pd.Series:
    return pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]


def _unique_feature_names(path: Path) -> list[str]:
    names = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 1].astype(str)
    # Keep the original names readable while making duplicate headers unambiguous.
    suffix = names.groupby(names, sort=False).cumcount()
    return [f"{name}_{number}" if number else name for name, number in zip(names, suffix)]


def load_dataset(dataset_dir: Path) -> HARData:
    """Load the predefined subject-disjoint train/test split from the UCI text files."""

    dataset_dir = Path(dataset_dir)
    required = [
        "features.txt",
        "activity_labels.txt",
        "train/X_train.txt",
        "train/y_train.txt",
        "train/subject_train.txt",
        "test/X_test.txt",
        "test/y_test.txt",
        "test/subject_test.txt",
    ]
    missing = [name for name in required if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Dataset is incomplete at {dataset_dir}. Missing: {', '.join(missing)}. "
            "Run `har-download` or provide the original UCI HAR Dataset directory."
        )

    feature_names = _unique_feature_names(dataset_dir / "features.txt")
    label_frame = pd.read_csv(
        dataset_dir / "activity_labels.txt", sep=r"\s+", header=None, index_col=0
    )
    label_names = label_frame.iloc[:, 0].astype(str).to_dict()

    def read_features(relative: str) -> pd.DataFrame:
        return pd.read_csv(dataset_dir / relative, sep=r"\s+", header=None, names=feature_names)

    X_train = read_features("train/X_train.txt")
    X_test = read_features("test/X_test.txt")
    raw_y_train = _read_vector(dataset_dir / "train/y_train.txt")
    raw_y_test = _read_vector(dataset_dir / "test/y_test.txt")
    y_train = raw_y_train.map(label_names).rename("activity")
    y_test = raw_y_test.map(label_names).rename("activity")
    subject_train = _read_vector(dataset_dir / "train/subject_train.txt").rename("subject")
    subject_test = _read_vector(dataset_dir / "test/subject_test.txt").rename("subject")

    if X_train.shape[1] != EXPECTED_FEATURE_COUNT or X_test.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features; found train={X_train.shape[1]}, "
            f"test={X_test.shape[1]}. Check that you supplied the original UCI dataset."
        )
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Unrecognized activity IDs found in y_train.txt or y_test.txt.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label row counts do not match.")
    if len(subject_train) != len(y_train) or len(subject_test) != len(y_test):
        raise ValueError("Subject and label row counts do not match.")
    if len(label_names) != EXPECTED_ACTIVITIES:
        raise ValueError(f"Expected {EXPECTED_ACTIVITIES} activity labels, found {len(label_names)}.")

    overlap = set(subject_train.unique()) & set(subject_test.unique())
    if overlap:
        raise ValueError(f"Official split contains subjects on both sides: {sorted(overlap)}")

    return HARData(X_train, y_train, subject_train, X_test, y_test, subject_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the UCI HAR dataset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    args = parser.parse_args()
    download_dataset(args.raw_dir, args.zip_path)


if __name__ == "__main__":
    main()
