# Human Activity Recognition on UCI-HAR

PCA-based comparison of seven classifiers (including a subject-aware RF + XGBoost + Logistic Regression stack) across full features and 95%, 99%, and 99.5% variance settings. Uses the UCI subject-disjoint train/test split and five-fold `GroupKFold` evaluation. Includes accuracy, macro-F1, confusion matrices, training/inference timing, model size, exploratory analysis, and a Streamlit PCA-space demo.

## Quick start

Python 3.10+ is required. Run from the project root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux:        source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
har-download
har-explore
har-train
streamlit run app.py
```

Training runs grouped cross-validation and fits multiple models, so allow several minutes or more depending on the computer. Reports are written to `reports/`; fitted pipelines are written to `models/`. See [GUIDE.md](GUIDE.md) for the detailed code walkthrough, exact result files, batch prediction, limitations, and troubleshooting.

The Streamlit page works with an official test sample or a CSV containing the 561 engineered UCI features. It does not read raw live phone sensors.

Dataset: [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). The downloaded data and generated models/reports are excluded from Git.
