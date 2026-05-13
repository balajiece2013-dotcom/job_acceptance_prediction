import pandas as pd
import numpy as np

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../data/job_data.csv")

print("Initial Shape:", df.shape)

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# REMOVE DUPLICATES
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# FIX INCONSISTENT VALUES (SAFE)
# -------------------------------
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

# -------------------------------
# OPTIONAL: STANDARDIZE GENDER
# -------------------------------
if "gender" in df.columns:
    df["gender"] = df["gender"].replace({
        "m": "male",
        "f": "female"
    })

# -------------------------------
# SAVE CLEANED DATA
# -------------------------------
df.to_csv("../data/cleaned_data.csv", index=False)

print("Cleaned Shape:", df.shape)
print("Data Cleaning Completed ✅")