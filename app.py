"""Simple UCI-HAR activity lab and model comparison dashboard."""

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

st.set_page_config(
    page_title="UCI-HAR Activity Lab",
    page_icon="📱",
    layout="wide",
)

model_files = sorted(MODEL_DIR.glob("*/*.joblib"))
assets_path = MODEL_DIR / "demo_assets.joblib"
if not model_files or not assets_path.is_file():
    st.error("Trained model files are missing. Run training from the project root first.")
    st.stop()

assets = joblib.load(assets_path)
representations = sorted({path.parent.name for path in model_files})


def representation_label(value: str) -> str:
    labels = {
        "no_pca": "Full features (561)",
        "pca_95": "PCA (95%)",
        "pca_99": "PCA (99%)",
        "pca_99_5": "PCA (99.5%)",
    }
    return labels.get(value, value.replace("_", " ").upper())


st.title("Smartphone Activity Recognition")
st.caption(
    "Try an activity prediction or compare the models and PCA settings."
)

control_cols = st.columns(2)
with control_cols[0]:
    representation = st.selectbox(
        "Feature representation",
        representations,
        format_func=representation_label,
    )
with control_cols[1]:
    model_names = sorted(
        path.stem for path in model_files if path.parent.name == representation
    )
    model_name = st.selectbox(
        "Classifier",
        model_names,
        format_func=lambda value: value.replace("_", " ").title(),
    )

bundle = joblib.load(MODEL_DIR / representation / f"{model_name}.joblib")
input_dimensions = (
    bundle["pca_components"] if bundle["pca"] is not None else 561
)
st.caption(
    f"Selected: {representation_label(representation)} · "
    f"{model_name.replace('_', ' ').title()} · {input_dimensions} input dimensions"
)

activity_tab, compare_tab = st.tabs(["Activity Lab", "Compare Models"])

with activity_tab:
    input_mode = st.radio(
        "Choose an input",
        ["Official test sample", "Upload feature CSV"],
        horizontal=True,
    )
    sample_label = None
    X = None

    if input_mode == "Official test sample":
        dataset_dir = ROOT / "data/raw/UCI HAR Dataset"
        try:
            if not (dataset_dir / "features.txt").is_file():
                dataset_dir = download_dataset(
                    ROOT / "data/raw", ROOT / "data/raw/uci_har.zip"
                )
            data = load_dataset(dataset_dir)
        except Exception as exc:
            st.error(f"Could not load the UCI HAR test set: {exc}")
        else:
            test_labels = data.y_test.reset_index(drop=True).astype(str)
            sample_key = "official_test_sample_index"
            if sample_key not in st.session_state:
                st.session_state[sample_key] = 0

            def choose_random_test_row():
                st.session_state[sample_key] = int(np.random.randint(len(data.X_test)))

            row_col, random_col = st.columns([4, 1])
            with row_col:
                sample_index = st.number_input(
                    "Test row (0-based)",
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
            st.caption("The actual activity is hidden until you reveal it.")
            X = data.X_test.iloc[[int(sample_index)]]
            sample_label = str(test_labels.iloc[int(sample_index)])
    else:
        uploaded = st.file_uploader(
            "Upload a CSV with the original UCI feature names", type=["csv"]
        )
        if uploaded is None:
            st.info("Upload one or more feature rows to get predictions.")
        else:
            X = pd.read_csv(uploaded)
            expected = bundle["feature_names"]
            missing = [name for name in expected if name not in X.columns]
            if missing:
                st.error(
                    f"CSV is missing {len(missing)} required feature columns. "
                    f"Examples: {missing[:5]}"
                )
                X = None
            else:
                X = X.loc[:, expected].apply(pd.to_numeric, errors="coerce")
                if X.empty or X.isna().any().any():
                    st.error("The uploaded CSV is empty or contains missing or nonnumeric values.")
                    X = None

    if X is not None:
        X = X.loc[:, bundle["feature_names"]]
        scaled = bundle["scaler"].transform(X)
        transformed = (
            bundle["pca"].transform(scaled) if bundle["pca"] is not None else scaled
        )
        predictions = bundle["model"].predict(transformed)
        probabilities = bundle["model"].predict_proba(transformed)
        classes = list(bundle["model"].classes_)
        first_prediction = str(predictions[0])
        top_probability = float(np.max(probabilities[0]))

        result_cols = st.columns(3)
        result_cols[0].metric(
            "Predicted activity", first_prediction.replace("_", " ").title()
        )
        result_cols[1].metric("Top-class probability", f"{top_probability:.1%}")
        result_cols[2].metric("Features used", str(input_dimensions))
        st.caption(
            "These are model-reported probabilities and have not been evaluated for "
            "calibration. A 100% score does not guarantee a correct prediction."
        )

        if sample_label is not None:
            with st.expander("Reveal the true activity"):
                actual_activity = sample_label.replace("_", " ").title()
                st.write(f"Actual activity: **{actual_activity}**")
                if first_prediction == sample_label:
                    st.success("The prediction matches the test label.")
                else:
                    st.warning("The prediction does not match the test label.")

        probability_frame = pd.DataFrame(
            {"Activity": classes, "Probability": probabilities[0]}
        ).sort_values("Probability")
        probability_figure = px.bar(
            probability_frame,
            x="Probability",
            y="Activity",
            orientation="h",
            labels={"Probability": "Model-reported probability"},
            title="Activity scores",
            color="Probability",
            color_continuous_scale="teal",
        )
        probability_figure.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(probability_figure, use_container_width=True)

        with st.expander("Show the sample in the 2D PCA plot"):
            embedding_frame = pd.DataFrame(
                {
                    "PC1": assets["train_embedding"][:, 0],
                    "PC2": assets["train_embedding"][:, 1],
                    "Activity": assets["train_labels"],
                }
            )
            pca_figure = px.scatter(
                embedding_frame,
                x="PC1",
                y="PC2",
                color="Activity",
                opacity=0.45,
                title="Training samples and selected input",
            )
            sample_xy = assets["pca"].transform(assets["scaler"].transform(X))
            pca_figure.add_trace(
                go.Scatter(
                    x=sample_xy[:, 0],
                    y=sample_xy[:, 1],
                    mode="markers",
                    name="Selected input",
                    marker={
                        "size": 14,
                        "color": "#111827",
                        "symbol": "star",
                        "line": {"width": 2, "color": "white"},
                    },
                )
            )
            st.plotly_chart(pca_figure, use_container_width=True)
            st.caption(
                "This two-dimensional plot is for visualization. The selected model "
                "uses the number of features shown above."
            )

        if len(X) > 1:
            output = pd.DataFrame({"predicted_activity": predictions})
            output = pd.concat(
                [
                    output,
                    pd.DataFrame(
                        probabilities,
                        columns=[f"probability_{value}" for value in classes],
                    ),
                ],
                axis=1,
            )
            st.subheader("Predictions for uploaded rows")
            st.dataframe(output, use_container_width=True, hide_index=True)
            st.download_button(
                "Download predictions",
                output.to_csv(index=False),
                "predictions.csv",
                mime="text/csv",
            )

    with st.expander("Participant robustness"):
        subject_path = REPORT_DIR / "subject_performance.csv"
        if not subject_path.is_file():
            st.info(
                "Generate the report after training with "
                "python -m har_project.subject_report."
            )
        else:
            subject_data = pd.read_csv(subject_path)
            selected_subjects = subject_data[
                (subject_data["representation"] == representation)
                & (subject_data["model"] == model_name)
            ].sort_values("subject_id")
            if selected_subjects.empty:
                st.info("No participant scores were found for this model.")
            else:
                stats = st.columns(3)
                stats[0].metric(
                    "Mean participant macro-F1",
                    f"{selected_subjects['macro_f1'].mean():.1%}",
                )
                stats[1].metric(
                    "Lowest participant macro-F1",
                    f"{selected_subjects['macro_f1'].min():.1%}",
                )
                stats[2].metric(
                    "Participants tested", str(len(selected_subjects))
                )
                subject_figure = px.bar(
                    selected_subjects,
                    x="subject_id",
                    y="macro_f1",
                    hover_data=["accuracy", "balanced_accuracy", "sample_count"],
                    labels={
                        "subject_id": "Held-out participant",
                        "macro_f1": "Macro-F1",
                    },
                    title="Score by held-out participant",
                )
                subject_figure.update_yaxes(range=[0, 1])
                st.plotly_chart(subject_figure, use_container_width=True)
                st.caption(
                    "This shows how performance varies across people excluded from training."
                )

with compare_tab:
    comparison_path = REPORT_DIR / "model_comparison.csv"
    if not comparison_path.is_file():
        st.info("Run training to create reports/model_comparison.csv.")
    else:
        comparison = pd.read_csv(comparison_path)
        chosen_representation = st.selectbox(
            "Compare representation",
            ["All representations"] + sorted(comparison["representation"].unique()),
            format_func=lambda value: (
                value if value == "All representations" else representation_label(value)
            ),
        )
        if chosen_representation != "All representations":
            comparison = comparison[
                comparison["representation"] == chosen_representation
            ]

        best_row = comparison.loc[comparison["macro_f1"].idxmax()]
        fastest_row = comparison.loc[
            comparison["inference_latency_ms_per_sample"].idxmin()
        ]
        compact_row = comparison.loc[comparison["model_size_mb"].idxmin()]
        summary_cols = st.columns(3)
        summary_cols[0].metric(
            "Best macro-F1",
            f"{best_row['macro_f1']:.1%}",
            f"{best_row['model']} · {best_row['representation']}",
        )
        summary_cols[1].metric(
            "Fastest inference",
            f"{fastest_row['inference_latency_ms_per_sample']:.4f} ms/sample",
            f"{fastest_row['model']} · {fastest_row['representation']}",
        )
        summary_cols[2].metric(
            "Smallest model",
            f"{compact_row['model_size_mb']:.2f} MB",
            f"{compact_row['model']} · {compact_row['representation']}",
        )

        score_figure = px.bar(
            comparison,
            x="model",
            y="macro_f1",
            color="representation",
            barmode="group",
            labels={"model": "Classifier", "macro_f1": "Test macro-F1"},
            title="Test performance",
            hover_data=["accuracy", "pca_components", "retained_variance"],
        )
        score_figure.update_yaxes(range=[0, 1])
        st.plotly_chart(score_figure, use_container_width=True)

        tradeoff_figure = px.scatter(
            comparison,
            x="inference_latency_ms_per_sample",
            y="macro_f1",
            color="representation",
            symbol="model",
            size="model_size_mb",
            hover_name="model",
            labels={
                "inference_latency_ms_per_sample": "Inference latency (ms/sample)",
                "macro_f1": "Test macro-F1",
                "model_size_mb": "Model size (MB)",
            },
            title="Accuracy, speed, and size",
        )
        st.plotly_chart(tradeoff_figure, use_container_width=True)

        st.subheader("Detailed results")
        st.dataframe(
            comparison.sort_values("macro_f1", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
