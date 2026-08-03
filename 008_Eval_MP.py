# ==========================================================
# Mini Project
# Title: Complete Machine Learning Model Evaluation using
# Classification & Regression with Scikit-learn Pipeline
# ==========================================================

# ==========================================================
# Import Libraries
# ==========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Classification Model
from sklearn.linear_model import LogisticRegression

# Regression Model
from sklearn.linear_model import Ridge

# Classification Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Regression Metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
#                 CLASSIFICATION
# ==========================================================

print("=" * 60)
print("CLASSIFICATION")
print("=" * 60)

# ----------------------------------------------------------
# Create Classification Dataset
# ----------------------------------------------------------

classification_data = {
    "StudyHours": [5, 2, 8, 3, 7, 1, 6, 4, 9, 2, 8, 3, 7, 1, 9, 4],
    "Attendance": [80, 50, 90, 60, 85, 40, 75, 65, 95, 45, 88, 55, 82, 35, 96, 62],
    "PreviousMarks": [70, 40, 85, 55, 80, 35, 72, 60, 92, 38, 86, 50, 78, 32, 94, 58],
    "Result": [
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail"
    ]
}

classification_df = pd.DataFrame(classification_data)

# ----------------------------------------------------------
# Select Features and Target
# ----------------------------------------------------------

X = classification_df[["StudyHours", "Attendance", "PreviousMarks"]]
y = classification_df["Result"]

# ----------------------------------------------------------
# Split Dataset
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# Create Pipeline
# ----------------------------------------------------------

classification_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression())
])

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

classification_model.fit(X_train, y_train)

# ----------------------------------------------------------
# Predict Labels
# ----------------------------------------------------------

predictions = classification_model.predict(X_test)

# ----------------------------------------------------------
# Predict Probabilities
# ----------------------------------------------------------

probabilities = classification_model.predict_proba(X_test)

pass_probabilities = probabilities[:, 1]

# ----------------------------------------------------------
# Display Confusion Matrix
# ----------------------------------------------------------

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        predictions,
        labels=["Pass", "Fail"]
    )
)

# ----------------------------------------------------------
# Classification Evaluation
# ----------------------------------------------------------

print("\nAccuracy:",
      accuracy_score(y_test, predictions))

print("Precision:",
      precision_score(
          y_test,
          predictions,
          pos_label="Pass"
      ))

print("Recall:",
      recall_score(
          y_test,
          predictions,
          pos_label="Pass"
      ))

print("F1 Score:",
      f1_score(
          y_test,
          predictions,
          pos_label="Pass"
      ))

print("ROC-AUC:",
      roc_auc_score(
          y_test,
          pass_probabilities
      ))

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================================================
#                    REGRESSION
# ==========================================================

print("\n")
print("=" * 60)
print("REGRESSION")
print("=" * 60)

# ----------------------------------------------------------
# Create Regression Dataset
# ----------------------------------------------------------

regression_data = {
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "Attendance": [40,50,55,60,70,75,85,90,95,98],
    "PreviousMarks": [35,45,50,60,70,78,85,92,94,97],
    "FinalMarks": [40,50,55,65,75,82,90,96,98,100]
}

regression_df = pd.DataFrame(regression_data)

# ----------------------------------------------------------
# Select Features and Target
# ----------------------------------------------------------

X = regression_df[["StudyHours", "Attendance", "PreviousMarks"]]
y = regression_df["FinalMarks"]

# ----------------------------------------------------------
# Split Dataset
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

# ----------------------------------------------------------
# Create Pipeline
# ----------------------------------------------------------

regression_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

# ----------------------------------------------------------
# Train Model
# ----------------------------------------------------------

regression_model.fit(X_train, y_train)

# ----------------------------------------------------------
# Predict Values
# ----------------------------------------------------------

predictions = regression_model.predict(X_test)

# ----------------------------------------------------------
# Calculate Regression Metrics
# ----------------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)

# ----------------------------------------------------------
# Display Results
# ----------------------------------------------------------

print("\nPredicted Values")

print(predictions)

print("\nActual Values")

print(y_test.values)

print("\nMean Absolute Error (MAE):")

print(mae)

print("\nMean Squared Error (MSE):")

print(mse)

print("\nRoot Mean Squared Error (RMSE):")

print(rmse)

print("\nR² Score:")

print(r2)

print("\n")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)