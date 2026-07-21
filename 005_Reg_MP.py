import pandas as pd
import numpy as np

# Import train-test split function
from sklearn.model_selection import train_test_split

# Import Pipeline and StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Import Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# Create Sample Dataset
# ==========================================

data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Attendance": [40, 50, 55, 60, 70, 75, 85, 90, 95, 98],
    "PreviousMarks": [35, 45, 50, 60, 70, 78, 85, 92, 94, 97],
    "FinalMarks": [40, 50, 55, 65, 75, 82, 90, 96, 98, 100]
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)


# ==========================================
# Select Features and Target Variable
# ==========================================

# Input Features
X = df[["StudyHours", "Attendance", "PreviousMarks"]]

# Output Target
y = df["FinalMarks"]


# ==========================================
# Split Dataset into Training and Testing Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ==========================================
# Create Regression Pipelines
# ==========================================

models = {

    # Linear Regression Pipeline
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    # Ridge Regression Pipeline
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    # Lasso Regression Pipeline
    "Lasso Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, max_iter=10000))
    ])
}


# ==========================================
# Train and Evaluate Each Model
# ==========================================

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict using test data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Display model performance
    print("\n", name)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)