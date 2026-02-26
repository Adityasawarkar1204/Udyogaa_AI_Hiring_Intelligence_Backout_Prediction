# src/app.py

import streamlit as st
import pandas as pd
import joblib
from features import build_features
from matching import load_encoder, match_job_resume

MODEL_DIR = "src/models"


@st.cache_resource
def load_artifacts():
    preproc = joblib.load(f"{MODEL_DIR}/preprocessor.joblib")
    model = joblib.load(f"{MODEL_DIR}/best_model.joblib")
    encoder = load_encoder()
    return preproc, model, encoder


def compute_final_score(row, risk_prob):
    tech = row.get("technical_score", 50) / 100
    comm = row.get("communication_score", 50) / 100
    stability = row.get("stability_score", 1) / 2

    score = 0.4 * tech + 0.2 * comm + 0.2 * (1 - risk_prob) + 0.2 * stability
    return round(score * 100, 2)


def get_risk_category(risk):
    if risk < 0.2:
        return "Low Risk"
    elif risk < 0.5:
        return "Medium Risk"
    else:
        return "High Risk"


# ---------------- UI START ---------------- #

st.set_page_config(page_title="Udyogaa AI Prototype", layout="wide")

st.title("Udyogaa AI — Hiring Intelligence Prototype")
st.markdown(
    "AI-powered backout prediction, candidate scoring & job matching system."
)

preproc, model, encoder = load_artifacts()

df = pd.read_csv("data/udyogaa_synthetic.csv")

# ================= Candidate Risk Section ================= #

st.header("Candidate Risk & Scoring")

st.subheader("Sample Candidate Data")
st.dataframe(df.head())

candidate_id = st.number_input(
    "Select Candidate ID",
    min_value=int(df.candidate_id.min()),
    max_value=int(df.candidate_id.max()),
    value=int(df.candidate_id.min())
)

candidate = df[df.candidate_id == candidate_id].iloc[0]

# ---------- Prediction ---------- #

X, _ = build_features(pd.DataFrame([candidate]))
X_processed = preproc.transform(X)

prob_join = model.predict_proba(X_processed)[:, 1][0]
risk = 1 - prob_join

final_score = compute_final_score(candidate, risk)
risk_category = get_risk_category(risk)

st.markdown("---")

col1, col2, col3 = st.columns(3)

col1.metric("Backout Risk", f"{round(risk*100,1)}%")
col2.metric("Risk Category", risk_category)
col3.metric("Final Candidate Score", final_score)

st.caption(f"Joining Probability: {round(prob_join*100,1)}%")

# ---------- Business Insight ---------- #

if risk < 0.2:
    st.success("Strong joining intent detected. Candidate is stable and low risk.")
elif risk < 0.5:
    st.warning("Moderate joining uncertainty. Recommend proactive engagement.")
else:
    st.error("High backout probability. Review offer strategy or engagement plan.")

st.markdown("---")


# ================= AI Job Matching Section ================= #

st.header("AI Job Matching")

job_desc = st.text_area(
    "Enter Job Description",
    "Looking for Python backend developer with API experience."
)

resume_text = st.text_area(
    "Enter Candidate Resume Text",
    "Python REST API PostgreSQL Machine Learning"
)

match_score = match_job_resume(encoder, job_desc, resume_text)

st.metric("Semantic Matching Score", round(match_score, 3))

# ---------- Matching Interpretation ---------- #

if match_score > 0.7:
    st.success("Strong semantic alignment between job role and candidate profile.")
elif match_score > 0.4:
    st.warning("Moderate skill alignment. Some gap may exist.")
else:
    st.error("Low job-resume similarity. Significant skill mismatch detected.")

st.markdown("---")

st.markdown(
"""
### How This Prototype Works:
- Backout risk predicted using best-performing ML model (auto-selected via ROC-AUC)
- Candidate score combines technical performance, communication, stability & risk signals
- Job matching powered by sentence-transformer embeddings using semantic similarity
"""
)

