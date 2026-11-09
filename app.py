import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------
# Bluetooth Antenna Fault Classification - Streamlit Front End
# Loads the five trained pipelines from ./model.
# The app evaluates uploaded test data only; it does not retrain models.
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Antenna Fault Classification",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path(__file__).parent / "model"
TARGET_COLUMN = "BT Status"
CLASS_LABELS = ["Fault", "Normal"]
EXPECTED_FEATURES = [
    "Length",
    "Width",
    "Height",
    "Permittivity",
    "Conductivity",
    "Bend",
    "Feed",
    "S11",
    "VSWR",
    "Gain",
    "Efficiency",
    "Bandwidth",
    "epsilon_r",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine (SVM)": "svm.joblib",
}


@st.cache_resource
def load_model(model_path: str):
    """Load a previously trained model pipeline from disk."""
    return joblib.load(model_path)


def get_model_classes(model):
    """Get class labels from a fitted sklearn estimator or Pipeline."""
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        for estimator in reversed(list(model.named_steps.values())):
            if hasattr(estimator, "classes_"):
                return list(estimator.classes_)
    raise ValueError("Could not determine the model's trained class labels.")


def validate_uploaded_data(data: pd.DataFrame):
    """Validate required input feature and target columns."""
    if data.empty:
        return False, "The uploaded CSV contains no rows."

    required_columns = EXPECTED_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        return False, "Missing required column(s): " + ", ".join(missing)

    if data[TARGET_COLUMN].isna().any():
        return False, f"The target column '{TARGET_COLUMN}' contains blank values."

    allowed_labels = set(CLASS_LABELS)
    actual_labels = set(data[TARGET_COLUMN].astype(str).unique())
    unexpected = actual_labels - allowed_labels
    if unexpected:
        return (
            False,
            f"Unexpected target label(s): {', '.join(sorted(unexpected))}. "
            "Expected only 'Fault' and 'Normal'.",
        )

    try:
        numeric_features = data[EXPECTED_FEATURES].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError):
        return False, "One or more input features contain non-numeric values."

    if numeric_features.isna().any().any():
        return False, "The input feature columns contain missing values."

    return True, ""


def evaluate_model(model, X: pd.DataFrame, y_true: pd.Series):
    """Generate predictions and classification metrics; Fault is the positive class."""
    y_pred = model.predict(X)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, pos_label="Fault", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, pos_label="Fault", zero_division=0
        ),
        "F1 Score": f1_score(
            y_true, y_pred, pos_label="Fault", zero_division=0
        ),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    # All five trained pipelines in this assignment support predict_proba.
    if not hasattr(model, "predict_proba"):
        raise ValueError("This saved model does not support probability predictions.")

    probabilities = model.predict_proba(X)
    model_classes = get_model_classes(model)
    if "Fault" not in model_classes:
        raise ValueError("The saved model does not contain the 'Fault' class.")

    fault_index = model_classes.index("Fault")
    y_binary = (y_true.astype(str) == "Fault").astype(int)
    metrics["AUC"] = roc_auc_score(y_binary, probabilities[:, fault_index])

    return y_pred, metrics


def load_all_models():
    """Load all model artifacts and report any missing files."""
    loaded = {}
    missing = []

    for display_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename
        if not model_path.is_file():
            missing.append(str(model_path.relative_to(Path(__file__).parent)))
            continue
        loaded[display_name] = load_model(str(model_path))

    return loaded, missing


# ------------------------------- Header -------------------------------

st.title("📡 Bluetooth Antenna Fault Classification")
st.caption(
    "Machine Learning Assignment 2 | Compare five trained classifiers "
    "using an uploaded labeled test dataset."
)

st.markdown(
    """
    This application loads the saved machine-learning pipelines and evaluates
    them on your uploaded CSV file. **Models are not retrained in this app.**
    The positive class for Precision, Recall, F1 Score, and AUC is **Fault**.
    """
)

with st.sidebar:
    st.header("Evaluation Controls")
    uploaded_file = st.file_uploader(
        "Upload test dataset (CSV)",
        type=["csv"],
        help=(
            "Upload a labeled test CSV containing the 13 model features "
            "and the target column 'BT Status'."
        ),
    )
    st.divider()
    st.markdown("**Target:** `BT Status`")
    st.markdown("**Classes:** `Fault`, `Normal`")
    st.markdown("**Input features:** 13 numerical features")

# Check model artifacts before attempting evaluation.
models, missing_model_files = load_all_models()

if missing_model_files:
    st.error(
        "Some trained model files are missing. Add all five `.joblib` files "
        "inside the `model/` folder at the same level as `app.py`."
    )
    for missing_file in missing_model_files:
        st.write(f"- `{missing_file}`")
    st.stop()

if uploaded_file is None:
    st.info("Upload your labeled test CSV from the left panel to begin evaluation.")
    with st.expander("Required CSV columns"):
        st.code(", ".join(EXPECTED_FEATURES + [TARGET_COLUMN]))
    st.stop()

# ------------------------------- Read data ----------------------------

try:
    uploaded_data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the uploaded CSV: {exc}")
    st.stop()

is_valid, validation_message = validate_uploaded_data(uploaded_data)
if not is_valid:
    st.error(validation_message)
    st.stop()

X_test = uploaded_data[EXPECTED_FEATURES].apply(pd.to_numeric, errors="raise")
y_test = uploaded_data[TARGET_COLUMN].astype(str)

st.subheader("Test Dataset Overview")
overview_1, overview_2, overview_3 = st.columns(3)
overview_1.metric("Test samples", f"{len(uploaded_data):,}")
overview_2.metric("Input features", len(EXPECTED_FEATURES))
overview_3.metric("Target classes", y_test.nunique())

class_counts = y_test.value_counts().reindex(CLASS_LABELS, fill_value=0)
st.write("**Actual class distribution**")
st.bar_chart(class_counts)

with st.expander("Preview uploaded test data"):
    st.dataframe(uploaded_data.head(10), use_container_width=True)
    st.caption(f"Showing first 10 of {len(uploaded_data)} uploaded rows.")

# ------------------------------- Model selection ----------------------

st.subheader("Model Evaluation")
selected_model_name = st.selectbox(
    "Select a classification model",
    options=list(MODEL_FILES.keys()),
    index=1,
)

try:
    selected_model = models[selected_model_name]
    y_pred, selected_metrics = evaluate_model(selected_model, X_test, y_test)
except Exception as exc:
    st.error(f"Evaluation failed for {selected_model_name}: {exc}")
    st.stop()

metric_columns = st.columns(6)
metric_order = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
for column, metric_name in zip(metric_columns, metric_order):
    column.metric(metric_name, f"{selected_metrics[metric_name]:.4f}")

# Confusion matrix
st.markdown("#### Confusion Matrix")
cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
fig, ax = plt.subplots(figsize=(5.5, 4))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_LABELS,
)
display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"{selected_model_name} - Confusion Matrix")
fig.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

# Classification report
st.markdown("#### Classification Report")
report = classification_report(
    y_test,
    y_pred,
    labels=CLASS_LABELS,
    target_names=CLASS_LABELS,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.round(4), use_container_width=True)

# ------------------------------- Comparison ---------------------------

st.subheader("Comparison of All Five Models")
st.caption(
    "Each saved model is evaluated against the same uploaded test dataset. "
    "No model is retrained."
)

comparison_rows = []
comparison_errors = []

for model_name, model in models.items():
    try:
        _, metrics = evaluate_model(model, X_test, y_test)
        comparison_rows.append({"Model": model_name, **metrics})
    except Exception as exc:
        comparison_errors.append(f"{model_name}: {exc}")

if comparison_rows:
    comparison_df = pd.DataFrame(comparison_rows).set_index("Model")
    comparison_df = comparison_df[metric_order]
    st.dataframe(
        comparison_df.style.format("{:.4f}").highlight_max(
            axis=0, subset=metric_order
        ),
        use_container_width=True,
    )

    best_model = comparison_df["F1 Score"].idxmax()
    st.success(
        f"Highest F1 Score on this uploaded dataset: **{best_model}** "
        f"({comparison_df.loc[best_model, 'F1 Score']:.4f})."
    )

if comparison_errors:
    with st.expander("Model evaluation warnings"):
        for error in comparison_errors:
            st.warning(error)

st.caption(
    "Interpretation note: reported metrics describe only the uploaded test "
    "dataset. They do not by themselves establish performance on unseen "
    "real-world antenna measurements."
)
