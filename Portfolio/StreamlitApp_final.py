import os
import sys
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

warnings.simplefilter("ignore")

# =========================
# AWS + CONSTANTS (MATCH NOTEBOOK)
# =========================
AWS_BUCKET = "rylie-mayo-s3-bucket"
ENDPOINT_NAME = "final-logistic-pipeline-endpoint-auto-6"

MODEL_INFO = {
    "model_name": "final-project-Bucket-Logistic-Model",
    "endpoint": ENDPOINT_NAME,
    "bucket": AWS_BUCKET,
    "explainer": "shap_explainer.pkl",
    "explainer_key": "explainer/shap_explainer.pkl",
    "pipeline": "model.tar.gz",
    "pipeline_key": "sklearn-pipeline-deployment/model.tar.gz"
}

# =========================
# AWS SESSION
# =========================
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]

@st.cache_resource
def get_session():
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1"
    )

session = get_session()
sm_session = sagemaker.Session(boto_session=session)

# =========================
# FEATURES
# =========================
FEATURES = [
    "loan_amnt","term","int_rate","installment","grade","sub_grade",
    "emp_length","home_ownership","annual_inc","verification_status",
    "purpose","dti","delinq_2yrs","earliest_cr_line","open_acc",
    "pub_rec","revol_bal","revol_util","total_acc","mort_acc","pub_rec_bankruptcies"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_pipeline():
    s3 = session.client("s3")

    local_tar = os.path.join(tempfile.gettempdir(), MODEL_INFO["pipeline"])

    if not os.path.exists(local_tar):
        s3.download_file(
            MODEL_INFO["bucket"],
            MODEL_INFO["pipeline_key"],
            local_tar
        )

    extract_dir = os.path.join(tempfile.gettempdir(), "model_dir")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir)
        files = tar.getnames()

    model_file = [f for f in files if f.endswith(".pkl") or f.endswith(".joblib")][0]
    return joblib.load(os.path.join(extract_dir, model_file))

# =========================
# LOAD SHAP
# =========================
@st.cache_resource
def load_explainer():
    s3 = session.client("s3")
    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])

    if not os.path.exists(local_path):
        s3.download_file(
            MODEL_INFO["bucket"],
            MODEL_INFO["explainer_key"],
            local_path
        )

    return load(local_path)

# =========================
# PREDICT
# =========================
def call_model_api(df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    pred = predictor.predict(df)
    val = int(pd.DataFrame(pred).values[-1][0])

    return "Low Risk" if val == 0 else "High Risk"

# =========================
# UI
# =========================
st.title("Loan Risk Prediction App")

loan_amnt = st.number_input("Loan Amount", 0.0, 100000.0, 10000.0)
int_rate = st.number_input("Interest Rate", 0.0, 40.0, 12.0)

if st.button("Predict"):
    df = pd.DataFrame([[loan_amnt, " 36 months", int_rate, 300,
                        "B","B3","10+ years","RENT",75000,
                        "Verified","debt_consolidation",18,0,
                        "Jan-2010",8,0,12000,45,20,1,0]],
                      columns=FEATURES)

    result = call_model_api(df)
    st.success(result)

    # SHAP
    try:
        explainer = load_explainer()
        model = load_pipeline()

        X = model[:-1].transform(df)

        if sparse.issparse(X):
            X = X.toarray()

        shap_values = explainer(X)

        st.subheader("SHAP Explanation")
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP error: {e}")
