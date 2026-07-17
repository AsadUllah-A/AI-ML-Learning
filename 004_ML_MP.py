# ===========================
# Import Libraries
# ===========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)

# ===========================
# Create Dataset
# ===========================

data = {
    "StudyHours": [5, 2, 8, 3, 7, 1, 6, 4, 9, 2, 8, 3],
    "Attendance": [80, 50, 90, 60, 85, 40, 75, 65, 95, 45, 88, 55],
    "PreviousMarks": [70, 40, 85, 55, 80, 35, 72, 60, 92, 38, 86, 50],
    "Result": [
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail"
    ]
}

# Dictionary ko DataFrame me convert karna
df = pd.DataFrame(data)

print("\nDataset\n")
print(df)

# ===========================
# Convert Pass/Fail into 1/0
# ===========================

df["Result"] = df["Result"].map({
    "Pass": 1,
    "Fail": 0
})

# ===========================
# Input and Output
# ===========================

X = df[["StudyHours", "Attendance", "PreviousMarks"]]

y = df["Result"]

# ===========================
# Split Data
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===========================
# Logistic Regression
# ===========================

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr_prediction = lr.predict(X_test)

print("\nLogistic Regression Accuracy")
print(accuracy_score(y_test, lr_prediction))

# ===========================
# Decision Tree
# ===========================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_prediction = dt.predict(X_test)

print("\nDecision Tree Accuracy")
print(accuracy_score(y_test, dt_prediction))

# ===========================
# KNN
# ===========================

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

knn_prediction = knn.predict(X_test)

print("\nKNN Accuracy")
print(accuracy_score(y_test, knn_prediction))

# ===========================
# Random Forest
# ===========================

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

rf_prediction = rf.predict(X_test)

print("\nRandom Forest Accuracy")
print(accuracy_score(y_test, rf_prediction))

# ===========================
# Grid Search
# ===========================

parameters = {
    "n_neighbors": [1, 3, 5]
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    parameters,
    cv=3
)

grid.fit(X_train, y_train)

print("\nBest Parameter")
print(grid.best_params_)

# ===========================
# Regression Metrics Example
# ===========================

actual = np.array([10, 20, 30, 40])

predicted = np.array([12, 18, 29, 41])

print("\nMean Absolute Error")
print(mean_absolute_error(actual, predicted))

print("\nMean Squared Error")
print(mean_squared_error(actual, predicted))

print("\nR2 Score")
print(r2_score(actual, predicted))

# ===========================
# Simple Plot
# ===========================

plt.scatter(df["StudyHours"], df["PreviousMarks"])

plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")
plt.title("Study Hours vs Previous Marks")

plt.show()