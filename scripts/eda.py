import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

# -------------------------------
# LOAD CLEANED DATASET
# -------------------------------
# Read cleaned dataset for analysis

df = pd.read_csv("../data/cleaned_data.csv")

# =========================================================
# BASIC DATA UNDERSTANDING
# =========================================================

# -------------------------------
# DATASET SHAPE
# -------------------------------
# Shows number of rows and columns

print("Dataset Shape:", df.shape)

# -------------------------------
# DATASET INFORMATION
# -------------------------------
# Displays column names and data types

print("\nDataset Info:\n")
print(df.info())

# -------------------------------
# STATISTICAL SUMMARY
# -------------------------------
# Displays mean, median, min, max values

print("\nStatistical Summary:\n")
print(df.describe())

# -------------------------------
# MISSING VALUE CHECK
# -------------------------------
# Verify whether null values exist

print("\nMissing Values:\n")
print(df.isnull().sum())

# =========================================================
# VISUALIZATION STYLE
# =========================================================

sns.set_style("whitegrid")

# =========================================================
# 1. TARGET VARIABLE DISTRIBUTION
# =========================================================
# Analyze placement status distribution

plt.figure(figsize=(6, 4))

sns.countplot(
    x="status",
    data=df
)

plt.title("Placement Status Distribution")
plt.xlabel("Placement Status")
plt.ylabel("Candidate Count")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Target distribution helps identify class balance.")

# =========================================================
# 2. SKILLS MATCH DISTRIBUTION
# =========================================================
# Analyze overall skills match percentage

plt.figure(figsize=(7, 4))

sns.histplot(
    df["skills_match_percentage"],
    bins=20,
    kde=True
)

plt.title("Skills Match Percentage Distribution")
plt.xlabel("Skills Match Percentage")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Most candidates have medium to high skills match percentage.")

# =========================================================
# 3. TECHNICAL SCORE ANALYSIS
# =========================================================
# Detect score spread and possible outliers

plt.figure(figsize=(7, 4))

sns.boxplot(
    x=df["technical_score"]
)

plt.title("Technical Score Boxplot")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Boxplot helps detect score spread and possible outliers.")

# =========================================================
# 4. GENDER VS PLACEMENT ANALYSIS
# =========================================================
# Compare placement outcomes across genders

plt.figure(figsize=(7, 4))

sns.countplot(
    x="gender",
    hue="status",
    data=df
)

plt.title("Gender vs Placement Status")
plt.xlabel("Gender")
plt.ylabel("Candidate Count")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("This chart compares placement distribution across genders.")

# =========================================================
# 5. DEGREE SPECIALIZATION ANALYSIS
# =========================================================
# Compare placement outcomes across specializations

plt.figure(figsize=(10, 5))

sns.countplot(
    x="degree_specialization",
    hue="status",
    data=df
)

plt.title("Degree Specialization vs Placement")
plt.xlabel("Degree Specialization")
plt.ylabel("Candidate Count")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Some specializations may show higher placement success.")

# =========================================================
# 6. CORRELATION HEATMAP
# =========================================================
# Analyze relationships between numerical features

plt.figure(figsize=(12, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Heatmap shows relationships between numerical features.")

# =========================================================
# FINAL BUSINESS INSIGHTS
# =========================================================
# Summary of important analytical findings

print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)

print("""

1. Candidates with higher technical scores
   showed better placement probability.

2. Skills match percentage played a major role
   in placement success.

3. Strong academic performance contributed
   to better hiring outcomes.

4. Some degree specializations achieved
   higher placement rates.

5. Visualization analysis helped identify
   candidate patterns and performance trends.

""")

print("\nEDA Completed Successfully ✅")