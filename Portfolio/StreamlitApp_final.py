import os
import sys
import io
import json
import warnings
import tempfile
import tarfile
import posixpath

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import joblib
import boto3
import sagemaker

from scipy import sparse
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
from joblib import load


# =========================
# Setup & Path Configuration
# =========================
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


# =========================
# Access secrets
# =========================
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]


# =========================
# AWS Session Management
# =========================
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)


# =========================
# Feature Config
# =========================
FEATURES = [
    "loan_amnt", "term", "int_rate", "installment",
    "grade", "sub_grade",
    "emp_length", "home_ownership",
    "annual_inc", "verification_status",
    "purpose", "dti",
    "delinq_2yrs", "earliest_cr_line",
    "open_acc", "pub_rec",
    "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies"
]

LABEL_MAP = {
    "loan_amnt": "Loan Amount",
    "term": "Loan Term",
    "int_rate": "Interest Rate",
    "installment": "Monthly Installment",
    "grade": "Loan Grade",
    "sub_grade": "Loan Sub-Grade",
    "emp_length": "Employment Length",
    "home_ownership": "Home Ownership",
    "annual_inc": "Annual Income",
    "verification_status": "Verification Status",
    "purpose": "Loan Purpose",
    "dti": "Debt-to-Income Ratio",
    "delinq_2yrs": "Delinquencies (Last 2 Years)",
    "earliest_cr_line": "Earliest Credit Line",
    "open_acc": "Open Credit Accounts",
    "pub_rec": "Public Records",
    "revol_bal": "Revolving Balance",
    "revol_util": "Revolving Utilization",
    "total_acc": "Total Credit Accounts",
    "mort_acc": "Mortgage Accounts",
    "pub_rec_bankruptcies": "Public Record Bankruptcies",
    "term_num": "Loan Term (Numeric)",
    "emp_length_num": "Employment Length (Numeric)",
    "int_rate_num": "Interest Rate (Numeric)",
    "revol_util_num": "Revolving Utilization (Numeric)",
    "loan_to_income": "Loan-to-Income Ratio",
    "revolving_burden": "Revolving Burden",
    "open_acc_ratio": "Open Account Ratio",
    "pub_rec_ratio": "Public Record Ratio",
    "dti_x_int_rate": "DTI × Interest Rate",
    "credit_age_per_account": "Credit Age per Account",
    "log_loan_amnt": "Log Loan Amount",
    "log_revol_bal": "Log Revolving Balance",
}

INPUTS = [
    {"name": "loan_amnt", "label": "Loan Amount", "kind": "number", "default": 10000.0, "min": 0.0, "step": 500.0},
    {"name": "term", "label": "Loan Term", "kind": "select", "options": [" 36 months", " 60 months"]},
    {"name": "int_rate", "label": "Interest Rate", "kind": "number", "default": 12.0, "min": 0.0, "step": 0.1},
    {"name": "installment", "label": "Monthly Installment", "kind": "number", "default": 300.0, "min": 0.0, "step": 10.0},
    {"name": "grade", "label": "Loan Grade", "kind": "select", "options": ["A", "B", "C", "D", "E", "F", "G"]},
    {"name": "sub_grade", "label": "Loan Sub-Grade", "kind": "text", "default": "B3"},
    {"name": "emp_length", "label": "Employment Length", "kind": "select", "options": [
        "< 1 year", "1 year", "2 years", "3 years", "4 years",
        "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
    ]},
    {"name": "home_ownership", "label": "Home Ownership", "kind": "select", "options": ["RENT", "OWN", "MORTGAGE", "OTHER", "ANY"]},
    {"name": "annual_inc", "label": "Annual Income", "kind": "number", "default": 75000.0, "min": 0.0, "step": 1000.0},
    {"name": "verification_status", "label": "Verification Status", "kind": "select", "options": ["Verified", "Source Verified", "Not Verified"]},
    {"name": "purpose", "label": "Loan Purpose", "kind": "select", "options": [
        "debt_consolidation", "credit_card", "home_improvement", "major_purchase",
        "small_business", "car", "medical", "moving", "vacation", "house", "wedding", "other"
    ]},
    {"name": "dti", "label": "Debt-to-Income Ratio", "kind": "number", "default": 18.0, "min": 0.0, "step": 0.1},
    {"name": "delinq_2yrs", "label": "Delinquencies (Last 2 Years)", "kind": "number", "default": 0.0, "min": 0.0, "step": 1.0},
    {"name": "earliest_cr_line", "label": "Earliest Credit Line", "kind": "text", "default": "Jan-2010"},
    {"name": "open_acc", "label": "Open Credit Accounts", "kind": "number", "default": 8.0, "min": 0.0, "step": 1.0},
    {"name": "pub_rec", "label": "Public Records", "kind": "number", "default": 0.0, "min": 0.0, "step": 1.0},
    {"name": "revol_bal", "label": "Revolving Balance", "kind": "number", "default": 12000.0, "min": 0.0, "step": 100.0},
    {"name": "revol_util", "label": "Revolving Utilization", "kind": "number", "default": 45.0, "min": 0.0, "step": 0.1},
    {"name": "total_acc", "label": "Total Credit Accounts", "kind": "number", "default": 20.0, "min": 0.0, "step": 1.0},
    {"name": "mort_acc", "label": "Mortgage Accounts", "kind": "number", "default": 1.0, "min": 0.0, "step": 1.0},
    {"name": "pub_rec_bankruptcies", "label": "Public Record Bankruptcies", "kind": "number", "default": 0.0, "min": 0.0, "step": 1.0},
]

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "shap_explainer.pkl",
    "pipeline": "model.tar.gz",
    "keys": FEATURES,
    "inputs": INPUTS,
    "label_map": LABEL_MAP
}


# =========================
# Helpers
# =========================
def clean_feature_name(name):
    name = str(name)
    for prefix in ["num__", "cat__", "remainder__"]:
        if name.startswith(prefix):
            name = name.replace(prefix, "")
    return MODEL_INFO["label_map"].get(name, name.replace("_", " ").title())


@st.cache_resource
def load_pipeline(_session, bucket, key_prefix):
    s3_client = _session.client("s3")
    filename = MODEL_INFO["pipeline"]
    local_tar_path = os.path.join(tempfile.gettempdir(), filename)

    if not os.path.exists(local_tar_path):
        s3_client.download_file(
            Bucket=bucket,
            Key=f"{key_prefix}/{os.path.basename(filename)}",
            Filename=local_tar_path
        )

    extract_dir = os.path.join(tempfile.gettempdir(), "model_extract_dir")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(local_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        joblib_files = [f for f in tar.getnames() if f.endswith(".joblib") or f.endswith(".pkl")]

    if not joblib_files:
        raise FileNotFoundError("No .joblib or .pkl file found inside model.tar.gz")

    model_path = os.path.join(extract_dir, joblib_files[0])
    return joblib.load(model_path)


@st.cache_resource
def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(Bucket=bucket, Key=key, Filename=local_path)

    with open(local_path, "rb") as f:
        return load(f)


# =========================
# Prediction Logic
# =========================
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]

        # Adjust this mapping if your model uses a different target encoding
        mapping = {
            0: "Low Risk / Fully Paid",
            1: "High Risk / Default"
        }

        return mapping.get(int(pred_val), f"Predicted Class: {pred_val}"), 200

    except Exception as e:
        return f"Error: {str(e)}", 500


# =========================
# Local Explainability
# =========================
def display_explanation(input_df, session, aws_bucket):
    try:
        explainer_name = MODEL_INFO["explainer"]
        explainer = load_shap_explainer(
            session,
            aws_bucket,
            posixpath.join("explainer", explainer_name),
            os.path.join(tempfile.gettempdir(), explainer_name)
        )

        best_pipeline = load_pipeline(session, aws_bucket, "sklearn-pipeline-deployment")

        # Manually mirror pipeline steps up to feature selection
        X_step = input_df.copy()
        X_step = best_pipeline.named_steps["cleaning"].transform(X_step)
        X_step = best_pipeline.named_steps["sanitization"].transform(X_step)
        X_step = best_pipeline.named_steps["feature_engineering"].transform(X_step)
        X_step = best_pipeline.named_steps["preprocessor"].transform(X_step)
        X_step = best_pipeline.named_steps["feature_selection"].transform(X_step)

        if sparse.issparse(X_step):
            X_step = X_step.toarray()

        try:
            feature_names_after_preprocessing = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        except Exception:
            feature_names_after_preprocessing = [
                f"feature_{i}" for i in range(X_step.shape[1])
            ]

        if hasattr(best_pipeline.named_steps["feature_selection"], "get_support"):
            selected_mask = best_pipeline.named_steps["feature_selection"].get_support()
            selected_features = feature_names_after_preprocessing[selected_mask]
        else:
            selected_features = feature_names_after_preprocessing

        pretty_selected_features = [clean_feature_name(f) for f in selected_features]
        input_df_transformed = pd.DataFrame(X_step, columns=pretty_selected_features)

        shap_values = explainer(input_df_transformed)

        st.subheader("Decision Transparency (SHAP)")

        fig = plt.figure(figsize=(10, 4))

        if len(shap_values.values.shape) == 3:
            shap_explanation = shap_values[0, :, 0]
        else:
            shap_explanation = shap_values[0]

        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

        top_feature = pd.Series(
            np.abs(shap_explanation.values),
            index=shap_explanation.feature_names
        ).idxmax()

        st.info(
            f"**Business Insight:** The most influential factor in this prediction was **{top_feature}**."
        )

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {repr(e)}")


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Loan Risk Prediction App", layout="wide")
st.title("Loan Risk Prediction App")
st.caption("Enter borrower and loan details to predict loan risk and view SHAP explanation.")

with st.form("pred_form"):
    st.subheader("Borrower & Loan Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            if inp["kind"] == "number":
                user_inputs[inp["name"]] = st.number_input(
                    inp["label"],
                    min_value=float(inp["min"]),
                    value=float(inp["default"]),
                    step=float(inp["step"])
                )
            elif inp["kind"] == "select":
                user_inputs[inp["name"]] = st.selectbox(
                    inp["label"],
                    inp["options"]
                )
            elif inp["kind"] == "text":
                user_inputs[inp["name"]] = st.text_input(
                    inp["label"],
                    value=inp["default"]
                )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    input_df = pd.DataFrame(
        [[user_inputs[k] for k in MODEL_INFO["keys"]]],
        columns=MODEL_INFO["keys"]
    )

    st.subheader("Input Summary")
    st.dataframe(input_df)

    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
