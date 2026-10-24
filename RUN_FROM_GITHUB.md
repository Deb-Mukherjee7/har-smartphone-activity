# Run the project on a new computer

This guide starts from the GitHub repository. Python 3.10 or newer and an internet connection are required. The dataset, trained models, and reports are generated locally and are not stored in GitHub.

## 1. Install prerequisites

Install Python 3.10+ and Git. Then open PowerShell on Windows or a terminal on macOS/Linux.

## 2. Clone the repository

```bash
git clone https://github.com/Deb-Mukherjee7/har-smartphone-activity.git
cd har-smartphone-activity
```

## 3. Create an environment and install packages

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is unavailable, use `python -m venv .venv` instead. Use Python 3.10 or newer.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Keep this environment active and run the remaining commands from the repository root. The requirements install the project in the environment, so these commands use the correct project copy.

## 4. Download the dataset and train

```bash
python -m har_project.data
python -m har_project.explore
python -m har_project.train
```

The downloader fetches and extracts the UCI HAR dataset. Training runs subject-wise cross-validation, fits the model comparisons, and can take several minutes or longer depending on the computer. Wait for the final `Saved comparison` and `Saved interactive demo model assets` messages.

## 5. Open the demo

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Keep the terminal open while using the app. Enter an official test row number or click **Random sample**; the prediction appears before the true label. Expand **Reveal the true activity** to check whether it was correct. You can also upload a CSV with the UCI engineered feature names. This demo does not collect live phone sensor data.

## 6. Find the results

- `reports/model_comparison.csv` — test and cross-validation scores, timing, and model sizes.
- `reports/confusion_matrices/` — per-model plots of correct and incorrect activity predictions.
- `reports/eda/` — exploratory analysis plots.
- `models/` — trained model files used by the demo.

These outputs are created on the computer and are ignored by Git.