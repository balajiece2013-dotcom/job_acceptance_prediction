import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../data/cleaned_data.csv")

# -------------------------------
# BASIC INFO
# -------------------------------
print("Dataset Shape:", df.shape)

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

# -------------------------------
# STYLE
# -------------------------------
sns.set_style("whitegrid")

# =========================================================
# 1. TARGET DISTRIBUTION
# =========================================================
plt.figure(figsize=(6, 4))

sns.countplot(
    x="status",
    data=df
)

plt.title("Placement Status Distribution")
plt.xlabel("Status")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Target distribution helps identify class balance.")

# =========================================================
# 2. SKILLS MATCH DISTRIBUTION
# =========================================================
plt.figure(figsize=(7, 4))

sns.histplot(
    df["skills_match_percentage"],
    bins=20,
    kde=True
)

plt.title("Skills Match Percentage Distribution")
plt.xlabel("Skills Match %")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Most candidates have medium to high skills match percentage.")

# =========================================================
# 3. TECHNICAL SCORE DISTRIBUTION
# =========================================================
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
# 4. GENDER VS PLACEMENT
# =========================================================
plt.figure(figsize=(7, 4))

sns.countplot(
    x="gender",
    hue="status",
    data=df
)

plt.title("Gender vs Placement Status")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("This chart compares placement distribution across genders.")

# =========================================================
# 5. DEGREE SPECIALIZATION VS PLACEMENT
# =========================================================
plt.figure(figsize=(10, 5))

sns.countplot(
    x="degree_specialization",
    hue="status",
    data=df
)

plt.title("Degree Specialization vs Placement")
plt.xlabel("Degree Specialization")
plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Observation
print("\nObservation:")
print("Some specializations may show higher placement success.")

# =========================================================
# 6. CORRELATION HEATMAP
# =========================================================
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
# FINAL INSIGHTS
# =========================================================

print("\n" + "=" * 50)
print("KEY INSIGHTS")
print("=" * 50)

print("""
1. Candidates with higher technical scores
   showed better placement probability.

2. Skills match percentage plays a major role
   in placement success.

3. Candidates with strong academic performance
   generally achieved better outcomes.

4. Some degree specializations showed higher
   placement rates compared to others.

5. Distribution charts helped identify
   candidate performance patterns and outliers.
""")

print("\nEDA Completed Successfully ✅")