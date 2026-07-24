import pandas as pd

# Import train-test split function
from sklearn.model_selection import train_test_split

# Import Pipeline and StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Import Evaluation Metric
from sklearn.metrics import accuracy_score


# ==========================================
# Create Sample Dataset
# ==========================================

data = {
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

# Convert dictionary into DataFrame
df = pd.DataFrame(data)


# ==========================================
# Select Features and Target Variable
# ==========================================

# Input Features
X = df[["StudyHours", "Attendance", "PreviousMarks"]]

# Target Variable
y = df["Result"]


# ==========================================
# Split Dataset into Training and Testing Data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==========================================
# Create Classification Models
# ==========================================

models = {

    # Logistic Regression with Feature Scaling
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ]),

    # K-Nearest Neighbors with Feature Scaling
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=3))
    ]),

    # Support Vector Machine with Feature Scaling
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="linear"))
    ]),

    # Gaussian Naive Bayes Classifier
    "Naive Bayes": GaussianNB(),

    # Decision Tree Classifier
    "Decision Tree": DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    ),

    # Random Forest Classifier
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        random_state=42
    ),

    # Gradient Boosting Classifier
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}


# ==========================================
# Train, Predict and Evaluate Each Model
# ==========================================

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on test data
    predictions = model.predict(X_test)

    # Calculate model accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Display model performance
    print(name, "Accuracy:", accuracy)