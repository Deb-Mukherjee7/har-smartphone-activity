# Detailed Project Guide

## 1. Project purpose

**Human Activity Recognition on UCI-HAR: PCA-Based Dimensionality Reduction with a Live Classification Demo** uses phone accelerometer and gyroscope features to classify six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying.

The synopsis asks whether principal component analysis (PCA) can substantially reduce the 561 input features while keeping classification performance close to full-feature models. It also asks for accuracy and macro-F1, class-level confusion, training time, inference latency, model size, unseen-subject evaluation, and an interactive demonstration. The participant report breaks held-out test scores down by person.

The system is designed as an educational and research benchmark. It does not access a phone, stream inertial signals, or derive UCI features from raw readings. The interactive app accepts engineered UCI feature rows or lets you choose a row from the official test set. The stretch goal of sensor-to-feature extraction requires additional phone-specific code and is outside this implementation.

## 2. Why PCA and why these models?

The UCI-HAR input has 561 features, many of which describe related properties of the same motion signals. PCA rotates standardized input into orthogonal components ordered by how much variance they explain. Keeping enough components to explain 95%, 99%, or 99.5% of training-set variance provides three compression settings to compare with using all 561 standardized features.

PCA is unsupervised: it preserves variance, not necessarily the distinctions that best separate activities. Therefore a higher retained-variance threshold is not guaranteed to yield better classification. The experiment measures that trade-off rather than assuming PCA improves the score.

The project compares:

| Classifier | Role in the comparison |
| --- | --- |
| Logistic Regression | Linear baseline |
| RBF Support Vector Machine | Nonlinear margin-based classifier |
| K-Nearest Neighbors | Distance-based baseline |
| Random Forest | Tree ensemble baseline |
| XGBoost | Gradient-boosted tree ensemble |
| MLP | Feed-forward neural baseline |
| Stacked ensemble | Random Forest + XGBoost + Logistic Regression, with a Logistic Regression meta-model |

Together these cover linear, kernel, distance-based, tree, boosting, neural, and stacked approaches requested in the synopsis. The fixed hyperparameters are in `src/har_project/features.py`; they are comparison settings, not a full hyperparameter search.

## 3. Dataset and leakage-aware evaluation

The code expects the original UCI Human Activity Recognition Using Smartphones archive: 10,299 windows, 561 features, 6 labels, and 30 subjects. It reads the official files `features.txt`, `activity_labels.txt`, and the train/test feature, label, and subject files.

The dataset has a predefined subject-disjoint split: roughly 21 subjects / 7,352 windows for training and 9 subjects / 2,947 windows for testing. `data.py` verifies that no subject ID occurs in both partitions.

Inside the official training portion, `GroupKFold` makes five validation folds based on subject ID. All windows from a person are kept together. In each outer fold, the scaler and PCA are fitted only on that fold's training subjects. The validation fold is then transformed and scored. After cross-validation, each model is fit using all official training subjects and scored on the held-out official test subjects.

The stacked ensemble has another grouped split inside each outer training fold. It creates out-of-fold class probabilities from RF, XGBoost, and Logistic Regression. The meta-model learns from those predictions, so each meta-training row comes from a subject the base learners did not train on. This avoids training the stack from in-sample predictions.

The official test set is evaluated for each model and representation because this project is a model comparison. Treat it as the final benchmark set: avoid repeatedly changing hyperparameters based on its scores. Use grouped cross-validation on the training subjects for future tuning.

## 4. Metrics and efficiency measurements

- **Accuracy:** fraction of all test windows classified correctly.
- **Macro-F1:** calculate F1 separately for each activity, then average all six equally. This makes errors on every activity visible even if class counts differ.
- **Balanced accuracy:** average recall across the six activity classes.
- **Confusion matrix:** rows show true labels and columns show predictions. Off-diagonal cells reveal which activities are mixed up, especially sitting versus standing.
- **Training time:** elapsed time to fit the final model on all official training rows; grouped cross-validation mean fit time is also included.
- **Inference latency:** average elapsed time per sample for repeated prediction batches of up to 256 official test rows. This is a local benchmark and varies by hardware and load.
- **Model size:** compressed serialized pipeline size on disk, including the fitted scaler, PCA if present, and classifier.

Latency and model size are measured on the Python/scikit-learn implementation, not on a phone or microcontroller. They are useful for relative comparison on the same machine.

## 5. Repository layout

```text
har-smartphone-activity/
├── app.py                         # Prediction and analysis dashboard
├── .streamlit/config.toml         # Light dashboard theme
├── config/default.yaml            # Paths, folds, PCA thresholds, model list
├── data/                          # Downloaded data (not committed)
├── models/                        # Saved pipelines and demo assets (not committed)
├── reports/                       # Metrics, plots, and comparison tables
├── src/har_project/
│   ├── data.py                    # Download, parse, and validate original UCI files
│   ├── explore.py                 # Class balance, missingness, correlations, PCA curve
│   ├── features.py                # Baselines and group-aware stacking classifier
│   ├── predict.py                 # Command-line batch prediction
│   ├── train.py                   # Grouped CV, final evaluation, reports, saved models
│   └── subject_report.py           # Held-out scores for each test participant
├── GUIDE.md                       # This detailed guide
├── README.md                      # Short quick start
├── pyproject.toml                 # Package metadata and command-line entry points
├── requirements.txt               # Editable project/dependency install
└── .gitignore                     # Excludes data, models, reports, and environments
```

### What each module does

#### `src/har_project/data.py`

Downloads the original UCI archive to `data/raw/`, checks ZIP paths before extraction, and loads the predefined split. It reads 561 feature headers and makes duplicate headers unique, maps numeric labels to readable activity names, reads subject IDs, and rejects missing or mismatched files and train/test subject overlap. The dataset files remain outside Git.

#### `src/har_project/explore.py`

Generates descriptive artifacts from training data: class counts, missing-value counts, a class-balance chart, a standardized PCA cumulative-variance curve with 95/99/99.5% markers, and a correlation matrix for the 30 highest-variance features. The scaler and PCA in this EDA step are fitted on training rows only.

#### `src/har_project/features.py`

Defines six baseline classifiers. XGBoost internally encodes the string activity labels to integers and maps predictions back to activity names. `GroupedStackingClassifier` generates base-model out-of-fold probabilities with `GroupKFold`, fits a logistic meta-model on those probabilities, then refits base models on the full input partition.

#### `src/har_project/train.py`

Runs four representations: all 561 standardized features, PCA retaining 95%, PCA retaining 99%, and PCA retaining 99.5% of variance. For each representation it runs grouped cross-validation for every configured classifier. Then it fits each classifier on all official training rows, evaluates on official test subjects, records efficiency measurements, saves its pipeline, and produces test confusion matrices. It also saves a separate 2D PCA embedding used only to visualize samples in the app.

The visualization PCA is not the same as the classifier PCA. It always has two components and is only used to display the sample location.

#### `src/har_project/subject_report.py`

Loads the saved models and evaluates each one on each participant in the official held-out test set. It writes per-participant accuracy, macro-F1, and balanced accuracy to `reports/subject_performance.csv`, plus mean, standard deviation, and lowest participant scores to `reports/subject_performance_summary.csv`. It does not refit models. Use these results as a robustness diagnostic, not to tune models against the held-out test set.

#### `src/har_project/predict.py`

Loads a saved comparison model and predicts from a headered CSV. It validates that all expected feature columns exist, orders them, applies the saved scaler/PCA, and writes labels and class probabilities to an output CSV.

#### `app.py`

Provides four dashboard sections: prediction, model comparison, participant robustness, and methodology. The user can select an anonymous official test row or a random sample, see the prediction and model-reported probability, then reveal the true label. Comparison charts show the accuracy/speed/size trade-off; the participant view shows score variation across held-out people. The app accepts UCI engineered feature rows or a compatible CSV; raw sensor streams are not accepted.

#### `config/default.yaml`

Stores the seed, data locations, 5-fold setting, PCA variance thresholds, list of models, and output directories. Edit this file for a different experiment. Do not change the official train/test subjects to a random row split.

## 6. Install and run

Run commands in a terminal from the folder containing `pyproject.toml`. Use Python 3.10 or newer. Downloading the dataset needs internet access. The complete grouped comparison fits many models, including nested models for stacking, so it can take several minutes or more depending on your computer.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
har-download
har-explore
har-train
python -m har_project.subject_report
streamlit run app.py
```

If activation is blocked, use Command Prompt and activate with `.venv\Scripts\activate.bat`.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
har-download
har-explore
har-train
python -m har_project.subject_report
streamlit run app.py
```

The browser opens the Streamlit page. Leave the terminal running while using the app. If an installed command is not found, use module commands:

```bash
python -m har_project.data
python -m har_project.explore
python -m har_project.train
python -m har_project.subject_report
```

Useful optional paths:

```bash
har-download --raw-dir data/raw --zip-path data/raw/uci_har.zip
har-train --config config/default.yaml --dataset-dir "data/raw/UCI HAR Dataset"
har-explore --dataset-dir "data/raw/UCI HAR Dataset" --output-dir reports/eda
```

## 7. How to inspect and check results

After `har-explore`:

| Output | How to use it |
| --- | --- |
| `reports/eda/class_balance.csv` and `.png` | Confirm class counts and relative balance |
| `reports/eda/missing_values_by_feature.csv` | Check missing values; the original features should be complete |
| `reports/eda/pca_explained_variance.csv` and `.png` | See how many PCs are needed to reach each target variance |
| `reports/eda/top30_feature_correlations.csv` and `.png` | Inspect redundancy among high-variance features |

After `har-train`:

| Output | How to use it |
| --- | --- |
| `reports/model_comparison.csv` | Main result table: test accuracy, macro-F1, balanced accuracy, grouped CV results, training time, inference latency, size, PCA component count, and retained variance |
| `reports/subject_performance.csv` | Accuracy, macro-F1, and balanced accuracy for every model and held-out participant |
| `reports/subject_performance_summary.csv` | Mean, standard deviation, and lowest participant score by model and representation |
| `reports/representation_summary.csv` | Quick representation-level score summary |
| `reports/confusion_matrices/` | One confusion matrix for each classifier and representation; inspect sitting/standing errors and per-class confusions |
| `reports/*_explained_variance.csv` | Exact retained PCA variance for each reduced representation |
| `reports/training_pca_2d.png` and `training_pca_2d.csv` | Training-set overview in two PCA dimensions |
| `reports/test_pca_2d.csv` | Held-out test subjects in the same 2D visualization coordinates |
| `reports/run_metadata.json` | Dataset sizes, subject IDs, and run configuration summary |
| `models/<representation>/<classifier>.joblib` | Saved fitted pipeline used by the demo or batch predictor |
| `models/demo_assets.joblib` | Fitted 2D PCA projection and points used by the demo |

For the core research question, compare `no_pca` with `pca_95`, `pca_99`, and `pca_99_5` for the same classifier. Look at macro-F1 and the confusion matrices alongside accuracy. Then compare PCA component count, training time, latency, and serialized size. Expected outcomes in the synopsis are goals to evaluate, not guaranteed scores.

`representation_summary.csv` is a convenience view; the detailed per-classifier rows in `model_comparison.csv` are the authoritative comparison.

## 8. Run batch predictions

Create a CSV with a header row using the original UCI feature names. Example:

```bash
har-predict my_feature_rows.csv \
  --model models/pca_99/stacked_ensemble.joblib \
  --output reports/my_predictions.csv
```

Use PowerShell line continuations with a backtick, or write the command on one line. The CSV can contain one or many rows. The output includes `predicted_activity` and probability columns for each class.

## 9. Interpreting limitations

- The UCI benchmark contains engineered 2.56-second windows, not raw sensor samples. True streaming requires matching the windowing, filtering, and feature-extraction pipeline.
- The selected PCA threshold preserves input variance, not class information. Accuracy can fall or occasionally improve after PCA.
- CPU time, model size, and latency depend on hardware and library versions; report the machine and environment when presenting measured results.
- The strict subject-disjoint test result may be lower than record-wise literature results. That is the expected cost of evaluating new-person generalization fairly.
- This is not a medical or safety system. Test across representative users and devices before deployment.
- The publication values in the synopsis provide context, but may use other datasets, splits, and preprocessing. They are not directly comparable unless protocols match.

## 10. Dataset attribution

Dataset: Anguita, D. et al., *A Public Domain Dataset for Human Activity Recognition Using Smartphones*, ESANN, 2013. UCI dataset page: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>. Check that page for current citation and license terms before redistributing the dataset.
