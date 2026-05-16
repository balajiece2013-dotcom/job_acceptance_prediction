import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# STREAMLIT DASHBOARD
# =========================================================

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="Job Acceptance Prediction System",

    layout="wide"

)

# =========================================================
# LOAD DATA & TRAINED MODEL
# =========================================================
# Load processed dataset and saved machine learning model

df = pd.read_csv("../data/final_data.csv")

model = joblib.load("../models/model.pkl")

model_columns = joblib.load("../models/columns.pkl")

# =========================================================
# DASHBOARD TITLE
# =========================================================

st.title("🎯 Job Acceptance Prediction System")

st.markdown(
    """
    This dashboard predicts whether a candidate is
    likely to accept a job offer based on academic,
    technical, and skill-related performance metrics.
    """
)

# =========================================================
# KPI SECTION
# =========================================================
# Display important business metrics

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

# Total candidates
col1.metric(

    "Total Candidates",

    len(df)

)

# Average skills percentage
col2.metric(

    "Avg Skills Match %",

    round(
        df["skills_match_percentage"].mean(),
        2
    )

)

# Average interview score
col3.metric(

    "Avg Interview Score",

    round(
        df["technical_score"].mean(),
        2
    )

)

# Acceptance rate
accept_rate = df["status"].mean() * 100

col4.metric(

    "Acceptance Rate %",

    round(
        accept_rate,
        2
    )

)

# =========================================================
# DATASET PREVIEW
# =========================================================

st.subheader("📁 Dataset Preview")

st.dataframe(df.head())

# =========================================================
# DATA VISUALIZATION SECTION
# =========================================================

st.subheader("📊 Data Visualizations")

# =========================================================
# ROW 1 VISUALIZATIONS
# =========================================================

col1, col2 = st.columns(2)

# ---------------------------------------------------------
# Placement Distribution
# ---------------------------------------------------------

with col1:

    fig1, ax1 = plt.subplots(
        figsize=(5, 3)
    )

    df["status"].value_counts().plot(

        kind="bar",

        ax=ax1

    )

    ax1.set_title(
        "Placement Distribution"
    )

    ax1.set_xlabel("Status")

    ax1.set_ylabel("Count")

    st.pyplot(fig1)

# ---------------------------------------------------------
# Skills Match Distribution
# ---------------------------------------------------------

with col2:

    fig2, ax2 = plt.subplots(
        figsize=(5, 3)
    )

    ax2.hist(

        df["skills_match_percentage"],

        bins=20

    )

    ax2.set_title(
        "Skills Match Distribution"
    )

    ax2.set_xlabel(
        "Skills Match %"
    )

    ax2.set_ylabel(
        "Frequency"
    )

    st.pyplot(fig2)

# =========================================================
# ROW 2 VISUALIZATIONS
# =========================================================

col3, col4 = st.columns(2)

# ---------------------------------------------------------
# Technical Score Boxplot
# ---------------------------------------------------------

with col3:

    fig3, ax3 = plt.subplots(
        figsize=(5, 3)
    )

    ax3.boxplot(

        df["technical_score"],

        vert=False

    )

    ax3.set_title(
        "Technical Score Boxplot"
    )

    st.pyplot(fig3)

# ---------------------------------------------------------
# Gender vs Placement
# ---------------------------------------------------------

with col4:

    gender_counts = pd.crosstab(

        df["gender"],

        df["status"]

    )

    fig4, ax4 = plt.subplots(
        figsize=(5, 3)
    )

    gender_counts.plot(

        kind="bar",

        ax=ax4

    )

    ax4.set_title(
        "Gender vs Placement"
    )

    ax4.set_xlabel("Gender")

    ax4.set_ylabel("Count")

    st.pyplot(fig4)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

st.subheader("🔥 Correlation Heatmap")

fig5, ax5 = plt.subplots(
    figsize=(10, 5)
)

# Correlation matrix
correlation = df.corr(
    numeric_only=True
)

# Heatmap
sns.heatmap(

    correlation,

    annot=True,

    fmt=".2f",

    cmap="coolwarm",

    linewidths=0.5,

    annot_kws={"size": 8},

    ax=ax5

)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(rotation=0)

st.pyplot(fig5)

# =========================================================
# SIDEBAR INPUT SECTION
# =========================================================
# User enters candidate information here

st.sidebar.header("🧾 Candidate Input")

# Age
age = st.sidebar.number_input(

    "Age",

    18,
    60,
    25

)

# Gender
gender = st.sidebar.selectbox(

    "Gender",

    df["gender"].unique()

)

# Academic scores
ssc = st.sidebar.number_input(

    "SSC Percentage",

    0.0,
    100.0,
    60.0

)

hsc = st.sidebar.number_input(

    "HSC Percentage",

    0.0,
    100.0,
    60.0

)

degree = st.sidebar.number_input(

    "Degree Percentage",

    0.0,
    100.0,
    60.0

)

# Degree specialization
specialization = st.sidebar.selectbox(

    "Degree Specialization",

    df["degree_specialization"].unique()

)

# Interview scores
tech_score = st.sidebar.number_input(

    "Technical Score",

    0.0,
    100.0,
    50.0

)

apt_score = st.sidebar.number_input(

    "Aptitude Score",

    0.0,
    100.0,
    50.0

)

comm_score = st.sidebar.number_input(

    "Communication Score",

    0.0,
    100.0,
    50.0

)

# Skills percentage
skills = st.sidebar.number_input(

    "Skills Match Percentage",

    0.0,
    100.0,
    50.0

)

# =========================================================
# PREDICTION SECTION
# =========================================================

if st.sidebar.button("Predict"):

    # -----------------------------------------------------
    # CREATE INPUT DATAFRAME
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # ENCODE INPUT DATA
    # -----------------------------------------------------

    input_encoded = pd.get_dummies(
        input_data
    )

    # Match training columns
    input_encoded = input_encoded.reindex(

        columns=model_columns,

        fill_value=0

    )

    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        input_encoded
    )[0]

    # Prediction probability
    prob = model.predict_proba(
        input_encoded
    )[0][1]

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.subheader("🎯 Prediction Result")

    # Accepted
    if prediction == 1:

        st.success(
            "✅ Candidate is likely to ACCEPT the job"
        )

    # Rejected
    else:

        st.error(
            "❌ Candidate is likely to REJECT the job"
        )

    # =====================================================
    # PROBABILITY ANALYSIS
    # =====================================================

    if prob > 0.7:

        st.success(

            f"🔥 High Acceptance Probability: "
            f"{round(prob * 100, 2)}%"

        )

    elif prob > 0.4:

        st.warning(

            f"⚠️ Medium Acceptance Probability: "
            f"{round(prob * 100, 2)}%"

        )

    else:

        st.error(

            f"❌ Low Acceptance Probability: "
            f"{round(prob * 100, 2)}%"

        )

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.subheader("📊 Feature Importance")

    importance = model.feature_importances_

    features = model_columns

    feat_df = pd.DataFrame({

        "feature": features,

        "importance": importance

    })

    # Sort features
    feat_df = feat_df.sort_values(

        by="importance",

        ascending=False

    )

    # Top 10 features
    feat_df = feat_df.head(10)

    # Reverse order for better display
    feat_df = feat_df[::-1]

    # Plot chart
    fig6, ax6 = plt.subplots(
        figsize=(8, 5)
    )

    ax6.barh(

        feat_df["feature"],

        feat_df["importance"]

    )

    ax6.set_xlabel(
        "Importance Score"
    )

    ax6.set_title(
        "Top 10 Important Features"
    )

    plt.tight_layout()

    st.pyplot(fig6)

# =========================================================
# END OF DASHBOARD
# =========================================================