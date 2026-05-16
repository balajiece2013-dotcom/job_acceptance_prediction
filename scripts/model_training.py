import pandas as pd
import joblib

# =========================================================
# JOB ACCEPTANCE PREDICTION SYSTEM
# MACHINE LEARNING MODEL TRAINING
# =========================================================

# -------------------------------
# IMPORT MACHINE LEARNING MODELS
# -------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

from sklearn.model_selection import train_test_split

# -------------------------------
# EVALUATION METRICS
# -------------------------------

from sklearn.metrics import (

    accuracy_score,
    classification_report,
    confusion_matrix

)

# =========================================================
# LOAD FINAL FEATURED DATASET
# =========================================================
# Read processed dataset for machine learning

df = pd.read_csv("../data/final_data.csv")

print("Dataset Shape:", df.shape)

# =========================================================
# DEFINE TARGET VARIABLE
# =========================================================
# status:
# 1 -> placed
# 0 -> not placed

target = "status"

# =========================================================
# DEFINE FEATURES & LABELS
# =========================================================

X = df.drop(columns=[target])

y = df[target]

# =========================================================
# ENCODE CATEGORICAL VARIABLES
# =========================================================
# Convert categorical columns into numerical format

X = pd.get_dummies(X)

# Save training column structure
# Used later during Streamlit prediction

joblib.dump(

    X.columns.tolist(),

    "../models/columns.pkl"

)

print("Categorical Encoding Completed ✅")

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)

print("Train-Test Split Completed ✅")

# =========================================================
# DEFINE MACHINE LEARNING MODELS
# =========================================================
# Multiple models are used for comparison

models = {

    "Logistic Regression":

    LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree":

    DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest":

    RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )
}

# =========================================================
# BEST MODEL TRACKING
# =========================================================
# Variables used to identify best-performing model

best_model = None

best_accuracy = 0

best_model_name = ""

# =========================================================
# MODEL TRAINING & EVALUATION
# =========================================================

for name, model in models.items():

    print("\n" + "=" * 50)

    print(f"MODEL: {name}")

    print("=" * 50)

    # -------------------------------
    # TRAIN MODEL
    # -------------------------------

    model.fit(
        X_train,
        y_train
    )

    # -------------------------------
    # MAKE PREDICTIONS
    # -------------------------------

    y_pred = model.predict(X_test)

    # -------------------------------
    # ACCURACY SCORE
    # -------------------------------

    accuracy = accuracy_score(

        y_test,
        y_pred

    )

    print(

        "\nAccuracy:",

        round(accuracy, 4)

    )

    # -------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------
    # Displays precision, recall, f1-score

    print("\nClassification Report:\n")

    print(

        classification_report(

            y_test,
            y_pred

        )

    )

    # -------------------------------
    # CONFUSION MATRIX
    # -------------------------------
    # Displays prediction performance summary

    print("\nConfusion Matrix:\n")

    print(

        confusion_matrix(

            y_test,
            y_pred

        )

    )

    # -------------------------------
    # STORE BEST MODEL
    # -------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name

# =========================================================
# SAVE BEST MODEL
# =========================================================
# Store best-performing model for deployment

joblib.dump(

    best_model,

    "../models/model.pkl"

)

print("\n" + "=" * 50)

print("BEST MODEL SAVED SUCCESSFULLY ✅")

print("=" * 50)

print("Best Model:", best_model_name)

print(

    "Best Accuracy:",

    round(best_accuracy, 4)

)

print("\nModel Training Completed Successfully ✅")