import pandas as pd

df = pd.read_csv("../data/cleaned_data.csv")

# -------------------------------
# TARGET COLUMN FIX (SAFE)
# -------------------------------
if "status" in df.columns:
    df["status"] = df["status"].str.strip().str.lower()

    df["status"] = df["status"].map({
        "placed": 1,
        "not placed": 0
    })

    # Handle unknown values
    df["status"] = df["status"].fillna(0)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# Experience Category (SAFE)
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

if "experience" in df.columns:
    df["exp_category"] = df["experience"].apply(exp_category)

# -------------------------------
# NEW FEATURES (IMPORTANT 🔥)
# -------------------------------

# Skills Level
if "skills_match_percentage" in df.columns:
    df["skills_level"] = pd.cut(
        df["skills_match_percentage"],
        bins=[0, 40, 70, 100],
        labels=["low", "medium", "high"]
    )

# Interview Performance
if "technical_score" in df.columns:
    df["interview_level"] = pd.cut(
        df["technical_score"],
        bins=[0, 40, 70, 100],
        labels=["poor", "average", "good"]
    )

# -------------------------------
# SAVE FINAL DATA
# -------------------------------
df.to_csv("../data/final_data.csv", index=False)

print("Feature Engineering Done ✅")