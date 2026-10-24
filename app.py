"""Interactive Streamlit demo for UCI-compatible feature rows."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from har_project.data import download_dataset, load_dataset

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

st.set_page_config(page_title="UCI-HAR PCA Demo", page_icon="🏃", layout="wide")
st.title("Smartphone Human Activity Recognition")
st.caption(
    "Choose an official test subject sample or upload UCI-compatible engineered features. "
    "This demo predicts from feature rows; it does not read raw phone sensors."
)

model_files = sorted(MODEL_DIR.glob("*/**/*.joblib"))
model_files = [path for path in model_files if path.name != "demo_assets.joblib"]
assets_path = MODEL_DIR / "demo_assets.joblib"
if not model_files or not assets_path.exists():
    st.error("Model artifacts are missing. Run `har-train` from the project root first.")
    st.stop()

assets = joblib.load(assets_path)
representations = sorted({path.parent.name for path in model_files})
left, right = st.columns(2)
with left:
    representation = st.selectbox("Feature representation", representations)
with right:
    model_names = sorted(path.stem for path in model_files if path.parent.name == representation)
    model_name = st.selectbox("Classifier", model_names)
bundle_path = MODEL_DIR / representation / f"{model_name}.joblib"
bundle = joblib.load(bundle_path)

input_mode = st.radio("Input", ["Choose a UCI test sample", "Upload feature CSV"], horizontal=True)
sample_label = None
if input_mode == "Choose a UCI test sample":
    dataset_dir = ROOT / "data/raw/UCI HAR Dataset"
    if not (dataset_dir / "features.txt").is_file():
        try:
            dataset_dir = download_dataset(ROOT / "data/raw", ROOT / "data/raw/uci_har.zip")
        except Exception as exc:
            st.error(f"Could not locate or download the UCI HAR dataset: {exc}")
            st.stop()
    data = load_dataset(dataset_dir)
    test_labels = data.y_test.reset_index(drop=True).astype(str)
    sample_key = "official_test_sample_index"
    if sample_key not in st.session_state:
        st.session_state[sample_key] = 0

    def choose_random_test_row():
        st.session_state[sample_key] = int(np.random.randint(len(data.X_test)))

    sample_col, random_col = st.columns([4, 1])
    with sample_col:
        sample_index = st.number_input(
            "Official test row (0-based)",
            min_value=0,
            max_value=len(data.X_test) - 1,
            step=1,
            key=sample_key,
        )
    with random_col:
        st.write("")
        st.button(
            "Random sample",
            on_click=choose_random_test_row,
            use_container_width=True,
        )
    st.caption("The true activity stays hidden until you choose to reveal it below.")

    X = data.X_test.iloc[[int(sample_index)]]
    sample_label = str(test_labels.iloc[int(sample_index)])
else:
    uploaded = st.file_uploader("Upload CSV with original UCI feature names", type=["csv"])
    if uploaded is None:
        st.info("Upload one or more feature rows with a header containing the UCI feature names.")
        st.stop()
    X = pd.read_csv(uploaded)
    expected = bundle["feature_names"]
    missing = [name for name in expected if name not in X.columns]
    if missing:
        st.error(f"CSV is missing {len(missing)} required feature columns. Example: {missing[:5]}")
        st.stop()
    X = X.loc[:, expected]
    if X.isna().any().any():
        st.error("The uploaded feature rows contain missing values.")
        st.stop()

expected = bundle["feature_names"]
X = X.loc[:, expected]
scaled = bundle["scaler"].transform(X)
transformed = bundle["pca"].transform(scaled) if bundle["pca"] is not None else scaled
predictions = bundle["model"].predict(transformed)
probabilities = bundle["model"].predict_proba(transformed)
classes = list(bundle["model"].classes_)
first_prediction = str(predictions[0])
confidence = float(np.max(probabilities[0]))

metric_cols = st.columns(3)
metric_cols[0].metric("Predicted activity", first_prediction.replace("_", " ").title())
metric_cols[1].metric("Model confidence", f"{confidence:.1%}")
metric_cols[2].metric(
    "PCA dimensions", str(bundle["pca_components"])
    if bundle["pca"] is not None
    else "561 (no PCA)"
)
if sample_label is not None:
    with st.expander("Reveal the true activity"):
        actual_activity = sample_label.replace("_", " ").title()
        st.write(f"Actual activity: **{actual_activity}**")
        if first_prediction == sample_label:
            st.success("The prediction matches the test label.")
        else:
            st.warning("The prediction does not match the test label.")
st.write(f"Selected: `{representation}` + `{model_name}`")

prob_df = pd.DataFrame({"Activity": classes, "Probability": probabilities[0]})
st.subheader("Class confidence")
st.bar_chart(prob_df.set_index("Activity"))

st.subheader("Sample in the 2D PCA space")
embedding_df = pd.DataFrame(
    {
        "PC1": assets["train_embedding"][:, 0],
        "PC2": assets["train_embedding"][:, 1],
        "Activity": assets["train_labels"],
    }
)
figure = px.scatter(
    embedding_df,
    x="PC1",
    y="PC2",
    color="Activity",
    opacity=0.45,
    title="Training subjects and selected sample",
)
sample_xy = assets["pca"].transform(assets["scaler"].transform(X))
figure.add_trace(
    go.Scatter(
        x=sample_xy[:, 0],
        y=sample_xy[:, 1],
        mode="markers",
        name="Selected sample",
        marker={"size": 15, "color": "black", "symbol": "star", "line": {"width": 2, "color": "white"}},
    )
)
st.plotly_chart(figure, use_container_width=True)
st.caption(
    "The displayed 2D coordinates come from a separate PCA visualization fitted only on "
    "training subjects. The classifier may use a different PCA component count."
)

st.subheader("Performance across held-out participants")
subject_report_path = REPORT_DIR / "subject_performance.csv"
if subject_report_path.is_file():
    subject_report = pd.read_csv(subject_report_path)
    selected_subjects = subject_report[
        (subject_report["representation"] == representation)
        & (subject_report["model"] == model_name)
    ].sort_values("subject_id")
    if selected_subjects.empty:
        st.info("No participant-level scores were found for this model.")
    else:
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "Mean participant macro-F1", f"{selected_subjects['macro_f1'].mean():.1%}"
        )
        metric_cols[1].metric(
            "Lowest participant macro-F1", f"{selected_subjects['macro_f1'].min():.1%}"
        )
        metric_cols[2].metric("Held-out participants", str(len(selected_subjects)))
        subject_figure = px.bar(
            selected_subjects,
            x="subject_id",
            y="macro_f1",
            hover_data=["accuracy", "balanced_accuracy", "sample_count"],
            labels={"subject_id": "Participant", "macro_f1": "Macro-F1"},
            title=f"{representation} / {model_name}",
        )
        subject_figure.update_yaxes(range=[0, 1])
        st.plotly_chart(subject_figure, use_container_width=True)
        st.caption(
            "Each bar scores one official test participant. Treat this as a robustness "
            "diagnostic; do not use held-out test scores to tune the models."
        )
else:
    st.info(
        "Generate participant-level scores after training with "
        "`python -m har_project.subject_report`."
    )

if len(X) > 1:
    st.subheader("Predictions for all uploaded rows")
    output = pd.DataFrame({"predicted_activity": predictions})
    output = pd.concat(
        [output, pd.DataFrame(probabilities, columns=[f"probability_{c}" for c in classes])],
        axis=1,
    )
    st.dataframe(output, use_container_width=True)
    st.download_button("Download predictions CSV", output.to_csv(index=False), "predictions.csv")
