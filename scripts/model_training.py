import pandas as pd
import joblib

# -------------------------------
# MODELS
# -------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
from sklearn.model_selection import train_test_split

# -------------------------------
# METRICS
# -------------------------------
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("../data/final_data.csv")

print("Dataset Shape:", df.shape)

# -------------------------------
# TARGET
# -------------------------------
target = "status"

# -------------------------------
# FEATURES & LABEL
# -------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# ENCODING
# -------------------------------
X = pd.get_dummies(X)

# Save training columns
joblib.dump(
    X.columns.tolist(),
    "../models/columns.pkl"
)

print("Encoding Completed ✅")

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-Test Split Done ✅")

# -------------------------------
# MODELS
# -------------------------------
models = {

    "Logistic Regression":
    LogisticRegression(max_iter=1000),

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

# -------------------------------
# BEST MODEL TRACKING
# -------------------------------
best_model = None
best_accuracy = 0
best_model_name = ""

# -------------------------------
# TRAIN & EVALUATE
# -------------------------------
for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"MODEL: {name}")
    print("=" * 50)

    # -------------------------------
    # TRAIN
    # -------------------------------
    model.fit(X_train, y_train)

    # -------------------------------
    # PREDICT
    # -------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------
    # ACCURACY
    # -------------------------------
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nAccuracy:",
          round(accuracy, 4))

    # -------------------------------
    # CLASSIFICATION REPORT
    # -------------------------------
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
    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # -------------------------------
    # SAVE BEST MODEL
    # -------------------------------
    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_model_name = name

# -------------------------------
# SAVE BEST MODEL
# -------------------------------
joblib.dump(
    best_model,
    "../models/model.pkl"
)

print("\n" + "=" * 50)
print("BEST MODEL SAVED ✅")
print("=" * 50)

print("Best Model:", best_model_name)

print(
    "Best Accuracy:",
    round(best_accuracy, 4)
)

print("\nModel Training Completed Successfully ✅")