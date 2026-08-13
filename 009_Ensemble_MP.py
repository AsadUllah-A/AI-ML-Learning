import pandas as pd

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Used to create ML pipelines
from sklearn.pipeline import Pipeline

# Standardizes numerical features
from sklearn.preprocessing import StandardScaler

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Ensemble learning models
from sklearn.ensemble import (
    VotingClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

# Used to calculate model accuracy
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# 1. CREATE DATASET
# ---------------------------------------------------------

data = {
    # Number of hours student studies
    "StudyHours": [5, 2, 8, 3, 7, 1, 6, 4, 9, 2, 8, 3, 7, 1, 9, 4],

    # Student attendance percentage
    "Attendance": [80, 50, 90, 60, 85, 40, 75, 65, 95, 45, 88, 55, 82, 35, 96, 62],

    # Marks from previous examination
    "PreviousMarks": [70, 40, 85, 55, 80, 35, 72, 60, 92, 38, 86, 50, 78, 32, 94, 58],

    # Target variable
    "Result": [
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail",
        "Pass", "Fail", "Pass", "Fail"
    ]
}


# Convert dictionary into a Pandas DataFrame
df = pd.DataFrame(data)


# ---------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

# Input features
X = df[["StudyHours", "Attendance", "PreviousMarks"]]

# Target/output
y = df["Result"]


# ---------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ---------------------------------------------------------

# 75% data is used for training
# 25% data is used for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,

    # Makes the split reproducible
    random_state=42,

    # Keeps Pass/Fail ratio similar in train and test data
    stratify=y
)


# ---------------------------------------------------------
# 4. LOGISTIC REGRESSION MODEL
# ---------------------------------------------------------

# Pipeline first scales the data,
# then applies Logistic Regression
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])


# ---------------------------------------------------------
# 5. K-NEAREST NEIGHBORS MODEL
# ---------------------------------------------------------

# KNN also benefits from feature scaling
# n_neighbors=3 means it checks 3 nearest data points
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=3))
])


# ---------------------------------------------------------
# 6. DECISION TREE MODEL
# ---------------------------------------------------------

# max_depth=3 limits tree depth
# This helps reduce overfitting
tree_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


# ---------------------------------------------------------
# 7. BASE MODELS FOR ENSEMBLE METHODS
# ---------------------------------------------------------

# Give names to the models
# These models will be used by Voting and Stacking
base_models = [
    ("logistic", logistic_model),
    ("knn", knn_model),
    ("tree", tree_model)
]


# ---------------------------------------------------------
# 8. ENSEMBLE MODELS
# ---------------------------------------------------------

models = {

    # -----------------------------------------------------
    # Voting Classifier
    # -----------------------------------------------------
    # Combines predictions from Logistic Regression,
    # KNN and Decision Tree
    # Soft voting uses prediction probabilities
    "Voting": VotingClassifier(
        estimators=base_models,
        voting="soft"
    ),


    # -----------------------------------------------------
    # Bagging Classifier
    # -----------------------------------------------------
    # Creates multiple Decision Trees using
    # different bootstrap samples of the dataset
    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),

        # Number of Decision Trees
        n_estimators=50,

        random_state=42
    ),


    # -----------------------------------------------------
    # Random Forest
    # -----------------------------------------------------
    # Creates many Decision Trees.
    # Random data samples and random features
    # are used to make the trees different.
    "Random Forest": RandomForestClassifier(
        n_estimators=100,

        # Maximum depth of each tree
        max_depth=3,

        random_state=42
    ),


    # -----------------------------------------------------
    # AdaBoost
    # -----------------------------------------------------
    # Builds weak models sequentially.
    # Each new model focuses more on previous mistakes.
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),

        # Number of weak learners
        n_estimators=50,

        # Contribution of each learner
        learning_rate=1.0,

        random_state=42
    ),


    # -----------------------------------------------------
    # Gradient Boosting
    # -----------------------------------------------------
    # Builds models sequentially.
    # Each new model tries to reduce previous errors.
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,

        # Learning rate controls the contribution
        # of each tree
        learning_rate=0.1,

        # Maximum depth of trees
        max_depth=3,

        random_state=42
    ),


    # -----------------------------------------------------
    # Stacking Classifier
    # -----------------------------------------------------
    # First layer contains Logistic Regression,
    # KNN and Decision Tree.
    # Final Logistic Regression combines their predictions.
    "Stacking": StackingClassifier(
        estimators=base_models,

        # Meta-model / final model
        final_estimator=LogisticRegression(),

        # Cross-validation used to train the final estimator
        cv=5
    )
}


# ---------------------------------------------------------
# 9. TRAIN AND TEST ALL MODELS
# ---------------------------------------------------------

# Loop through every ensemble model
for name, model in models.items():

    # Train model using training data
    model.fit(X_train, y_train)

    # Predict results for test data
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Display model name and accuracy
    print(name, "Accuracy:", accuracy)