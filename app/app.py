import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATA & MODEL
# -------------------------------
df = pd.read_csv("../data/final_data.csv")
model = joblib.load("../models/model.pkl")
model_columns = joblib.load("../models/columns.pkl")

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Job Acceptance Prediction", layout="wide")
st.title("🎯 Job Acceptance Prediction System")

# -------------------------------
# KPI SECTION
# -------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Candidates", len(df))
col2.metric("Avg Skills Match %", round(df["skills_match_percentage"].mean(), 2))
col3.metric("Avg Interview Score", round(df["technical_score"].mean(), 2))

accept_rate = df["status"].mean() * 100
col4.metric("Acceptance Rate %", round(accept_rate, 2))

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# VISUALIZATIONS
# -------------------------------
st.subheader("📊 Data Visualizations")

# Placement Distribution
fig1, ax1 = plt.subplots(figsize=(5, 4))

df["status"].value_counts().plot(
    kind="bar",
    ax=ax1
)

ax1.set_title("Placement Status Distribution")
ax1.set_xlabel("Status")
ax1.set_ylabel("Count")

st.pyplot(fig1)

# Skills Match Distribution
fig2, ax2 = plt.subplots(figsize=(6, 4))

ax2.hist(
    df["skills_match_percentage"],
    bins=20
)

ax2.set_title("Skills Match Distribution")
ax2.set_xlabel("Skills Match %")
ax2.set_ylabel("Frequency")

st.pyplot(fig2)

# Technical Score Boxplot
fig3, ax3 = plt.subplots(figsize=(6, 3))

ax3.boxplot(df["technical_score"], vert=False)

ax3.set_title("Technical Score Boxplot")

st.pyplot(fig3)

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("🧾 Candidate Input")

age = st.sidebar.number_input("Age", 18, 60, 25)
gender = st.sidebar.selectbox("Gender", df["gender"].unique())

ssc = st.sidebar.number_input("SSC %", 0.0, 100.0, 60.0)
hsc = st.sidebar.number_input("HSC %", 0.0, 100.0, 60.0)
degree = st.sidebar.number_input("Degree %", 0.0, 100.0, 60.0)

specialization = st.sidebar.selectbox(
    "Degree Specialization",
    df["degree_specialization"].unique()
)

tech_score = st.sidebar.number_input("Technical Score", 0.0, 100.0, 50.0)
apt_score = st.sidebar.number_input("Aptitude Score", 0.0, 100.0, 50.0)
comm_score = st.sidebar.number_input("Communication Score", 0.0, 100.0, 50.0)

skills = st.sidebar.number_input("Skills Match %", 0.0, 100.0, 50.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("Predict"):

    input_data = pd.DataFrame({
        "age_years": [age],
        "gender": [gender],
        "ssc_percentage": [ssc],
        "hsc_percentage": [hsc],
        "degree_percentage": [degree],
        "degree_specialization": [specialization],
        "technical_score": [tech_score],
        "aptitude_score": [apt_score],
        "communication_score": [comm_score],
        "skills_match_percentage": [skills]
    })

    # -------------------------------
    # ENCODING (MATCH TRAINING)
    # -------------------------------
    input_encoded = pd.get_dummies(input_data)

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    prediction = model.predict(input_encoded)[0]
    prob = model.predict_proba(input_encoded)[0][1]

    # -------------------------------
    # RESULT
    # -------------------------------
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success("✅ Candidate is likely to ACCEPT the job")
    else:
        st.error("❌ Candidate is likely to REJECT the job")

    # Probability display
    if prob > 0.7:
        st.success(f"🔥 High Acceptance Probability: {round(prob*100,2)}%")
    elif prob > 0.4:
        st.warning(f"⚠️ Medium Acceptance Probability: {round(prob*100,2)}%")
    else:
        st.error(f"❌ Low Acceptance Probability: {round(prob*100,2)}%")

    # -------------------------------
    # FEATURE IMPORTANCE (CLEAN GRAPH)
    # -------------------------------
    st.subheader("📊 Feature Importance")

    importance = model.feature_importances_
    features = model_columns

    feat_df = pd.DataFrame({
        "feature": features,
        "importance": importance
    })

    # Sort descending
    feat_df = feat_df.sort_values(by="importance", ascending=False)

    # Top 10 features only
    feat_df = feat_df.head(10)

    # Reverse for horizontal view
    feat_df = feat_df[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_df["feature"], feat_df["importance"])

    ax.set_xlabel("Importance Score")
    ax.set_title("Top 10 Important Features")

    plt.tight_layout()

    st.pyplot(fig)