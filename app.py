"""Small interactive Streamlit app for the diabetes course project."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
RESULTS_PATH = ROOT / "reports" / "final_model_results.json"

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


st.set_page_config(page_title="Diabetes Prediction Lab", page_icon="🩺", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    """Fit the notebook's selected model using the same fixed training split."""
    data = load_data()
    X = data[FEATURES].copy()
    for column in ZERO_MEANS_MISSING:
        X[column] = X[column].replace(0, np.nan)
    y = data["Outcome"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def clean_for_model(values):
    row = pd.DataFrame([values], columns=FEATURES)
    for column in ZERO_MEANS_MISSING:
        row[column] = row[column].replace(0, np.nan)
    return row


st.title("Diabetes Prediction Lab")
st.write(
    "Explore the Pima diabetes dataset and try different measurements with the "
    "Logistic Regression model from our project."
)
st.warning(
    "For learning and demonstration only. This model is not a medical device and "
    "cannot diagnose diabetes or guide treatment. The dataset contains historical "
    "records from one specific population."
)

prediction_tab, explore_tab, model_tab = st.tabs(
    ["Try a prediction", "Explore the dataset", "How the model works"]
)

with prediction_tab:
    st.subheader("Enter example measurements")
    st.caption(
        "Change the numbers and select **Estimate** to see how the model responds. "
        "For Glucose, Blood Pressure, Skin Thickness, Insulin, and BMI, entering 0 "
        "means the measurement is missing; the model fills it using its training median."
    )
    data = load_data()
    medians = data[FEATURES].copy()
    for column in ZERO_MEANS_MISSING:
        medians[column] = medians[column].replace(0, np.nan)
    medians = medians.median()

    with st.form("measurements"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input(
                "Pregnancies", min_value=0, max_value=20, value=int(medians["Pregnancies"]),
                help="Number of times pregnant. Zero is a valid value.",
            )
            glucose = st.number_input(
                "Glucose", min_value=0, max_value=300, value=int(medians["Glucose"]),
                help="Plasma glucose measurement. Enter 0 if it was not measured.",
            )
            blood_pressure = st.number_input(
                "Blood pressure", min_value=0, max_value=200,
                value=int(medians["BloodPressure"]),
                help="Diastolic blood pressure. Enter 0 if it was not measured.",
            )
            skin_thickness = st.number_input(
                "Skin thickness", min_value=0, max_value=120,
                value=int(medians["SkinThickness"]),
                help="Triceps skinfold thickness. Enter 0 if it was not measured.",
            )
        with col2:
            insulin = st.number_input(
                "Insulin", min_value=0, max_value=2000,
                value=int(medians["Insulin"]),
                help="2-hour serum insulin. Enter 0 if it was not measured.",
            )
            bmi = st.number_input(
                "BMI", min_value=0.0, max_value=100.0,
                value=float(medians["BMI"]), step=0.1,
                help="Body mass index. Enter 0 if it was not measured.",
            )
            pedigree = st.number_input(
                "Diabetes pedigree function", min_value=0.0, max_value=3.0,
                value=float(medians["DiabetesPedigreeFunction"]), step=0.01,
                help="Dataset score that summarizes family-history information.",
            )
            age = st.number_input(
                "Age (years)", min_value=21, max_value=120, value=int(medians["Age"]),
            )
        submitted = st.form_submit_button("Estimate from these measurements", type="primary")

    if submitted:
        values = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
        row = clean_for_model(values)
        model = load_model()
        probability = float(model.predict_proba(row)[0, 1])
        predicted_class = int(probability >= 0.5)

        st.divider()
        if predicted_class == 1:
            st.error("The model assigns this example to its higher estimated probability group.")
        else:
            st.success("The model assigns this example to its lower estimated probability group.")
        st.metric("Estimated probability for the dataset's diabetes label", f"{probability:.1%}")
        st.progress(probability)
        st.caption(
            "This percentage is the model's estimate for a record like this in the training "
            "dataset. It is not a personal medical risk estimate or a diagnosis."
        )

with explore_tab:
    data = load_data()
    st.subheader("The dataset used in this project")
    positive_rate = float(data["Outcome"].mean())
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Patient records", f"{len(data):,}")
    metric2.metric("Measurements", len(FEATURES))
    metric3.metric("Records labeled Outcome = 1", f"{positive_rate:.1%}")
    st.caption(
        "Outcome = 1 is the label recorded in the dataset. These records are from "
        "Pima women aged 21 or older, so the data does not represent everyone."
    )

    feature = st.selectbox("Choose a measurement to explore", FEATURES)
    bins = st.slider("Number of chart groups", min_value=5, max_value=40, value=20)
    chart_data = data[[feature, "Outcome"]].copy()
    if feature in ZERO_MEANS_MISSING:
        chart_data[feature] = chart_data[feature].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(
        data=chart_data,
        x=feature,
        hue="Outcome",
        bins=bins,
        multiple="layer",
        alpha=0.45,
        ax=ax,
    )
    ax.set_title(f"{feature} values, grouped by recorded outcome")
    ax.set_xlabel(feature)
    ax.set_ylabel("Number of records")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    with st.expander("Show some dataset rows"):
        st.dataframe(data.head(20), use_container_width=True)
    with st.expander("Show zero-coded missing measurements"):
        missing_counts = pd.DataFrame(
            {
                "Column": ZERO_MEANS_MISSING,
                "Zero values treated as missing": [int((data[col] == 0).sum()) for col in ZERO_MEANS_MISSING],
            }
        )
        st.dataframe(missing_counts, hide_index=True, use_container_width=True)

with model_tab:
    st.subheader("What the model does")
    st.markdown(
        "1. It learns patterns from the notebook's 80% training portion of the dataset.\n"
        "2. It fills missing measurements with medians learned from that training portion.\n"
        "3. It scales the measurements and applies Logistic Regression (C = 0.1).\n"
        "4. The separate 20% test portion was kept aside for the reported evaluation."
    )
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open(encoding="utf-8") as results_file:
            results = json.load(results_file)
        metrics = results["test_metrics"]
        st.subheader("Notebook's held-out test results")
        st.caption("These are the project's saved test results, not a score for the inputs above.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['Precision']:.1%}")
        m3.metric("Recall", f"{metrics['Recall']:.1%}")
        m4.metric("F1 score", f"{metrics['F1']:.1%}")
        m5.metric("ROC-AUC", f"{metrics['ROC_AUC']:.2f}")
        st.info(
            "For example, recall is about 48% on the held-out test records: the model "
            "found about 48 out of every 100 records labeled Outcome = 1 in that test set."
        )
