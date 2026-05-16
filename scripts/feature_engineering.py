import pandas as pd

# =========================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# FEATURE ENGINEERING
# =========================================================

# -------------------------------
# LOAD CLEANED DATASET
# -------------------------------
# Read cleaned dataset for feature creation

df = pd.read_csv("../data/cleaned_data.csv")

# =========================================================
# TARGET COLUMN TRANSFORMATION
# =========================================================
# Convert placement status into numerical labels
# placed      -> 1
# not placed  -> 0

if "status" in df.columns:

    # Standardize text values
    df["status"] = (
        df["status"]
        .str.strip()
        .str.lower()
    )

    # Convert categorical labels to numeric
    df["status"] = df["status"].map({

        "placed": 1,
        "not placed": 0

    })

    # Handle unknown or missing values
    df["status"] = df["status"].fillna(0)

# =========================================================
# EXPERIENCE CATEGORY FEATURE
# =========================================================
# Create experience levels based on years of experience

def exp_category(x):

    try:

        if x < 2:
            return "fresher"

        elif x < 5:
            return "junior"

        else:
            return "senior"

    except:

        return "fresher"

# Apply feature engineering
if "experience" in df.columns:

    df["exp_category"] = df["experience"].apply(
        exp_category
    )

# =========================================================
# SKILLS LEVEL FEATURE
# =========================================================
# Convert skills percentage into categories

if "skills_match_percentage" in df.columns:

    df["skills_level"] = pd.cut(

        df["skills_match_percentage"],

        bins=[0, 40, 70, 100],

        labels=[
            "low",
            "medium",
            "high"
        ]

    )

# =========================================================
# INTERVIEW PERFORMANCE FEATURE
# =========================================================
# Categorize technical interview scores

if "technical_score" in df.columns:

    df["interview_level"] = pd.cut(

        df["technical_score"],

        bins=[0, 40, 70, 100],

        labels=[
            "poor",
            "average",
            "good"
        ]

    )

# =========================================================
# SAVE FINAL FEATURED DATASET
# =========================================================
# Store processed dataset for model training

df.to_csv(
    "../data/final_data.csv",
    index=False
)

print("Feature Engineering Completed Successfully ✅")