import pandas as pd
import numpy as np

# =========================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# DATA CLEANING & PREPROCESSING
# =========================================================

# -------------------------------
# LOAD RAW DATASET
# -------------------------------
# Read the original dataset from CSV file

df = pd.read_csv("../data/job_data.csv")

print("Initial Dataset Shape:", df.shape)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

# -------------------------------
# HANDLE NUMERICAL MISSING VALUES
# -------------------------------
# Fill numerical null values using median
# Median is safer for outliers compared to mean

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:

    df[col] = df[col].fillna(
        df[col].median()
    )

# -------------------------------
# HANDLE CATEGORICAL MISSING VALUES
# -------------------------------
# Fill categorical null values using mode
# Mode represents the most frequent category

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )

# =========================================================
# REMOVE DUPLICATE RECORDS
# =========================================================
# Remove repeated rows to improve data quality

df = df.drop_duplicates()

# =========================================================
# FIX INCONSISTENT CATEGORICAL VALUES
# =========================================================
# Convert text columns to lowercase
# Remove extra spaces for consistency

for col in cat_cols:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

# =========================================================
# STANDARDIZE GENDER VALUES
# =========================================================
# Convert short forms into full labels

if "gender" in df.columns:

    df["gender"] = df["gender"].replace({

        "m": "male",
        "f": "female"

    })

# =========================================================
# SAVE CLEANED DATASET
# =========================================================
# Store cleaned data for further analysis

df.to_csv(
    "../data/cleaned_data.csv",
    index=False
)

print("Cleaned Dataset Shape:", df.shape)

print("Data Cleaning Completed Successfully ✅")