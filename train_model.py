import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("bugs.csv")

X = data["bug_description"]

# -----------------------------
# BUG TYPE MODEL
# -----------------------------

y_type = data["bug_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_type,
    test_size=0.2,
    random_state=42,
    stratify=y_type
)

type_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

type_model.fit(X_train, y_train)

type_predictions = type_model.predict(X_test)

type_accuracy = accuracy_score(
    y_test,
    type_predictions
)

print("Bug Type Model Accuracy:",
      round(type_accuracy * 100, 2), "%")


# -----------------------------
# SEVERITY MODEL
# -----------------------------

y_severity = data["severity"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_severity,
    test_size=0.2,
    random_state=42,
    stratify=y_severity
)

severity_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

severity_model.fit(X_train, y_train)

severity_predictions = severity_model.predict(X_test)

severity_accuracy = accuracy_score(
    y_test,
    severity_predictions
)

print("Severity Model Accuracy:",
      round(severity_accuracy * 100, 2), "%")


# Save models
joblib.dump(type_model, "bug_type_model.pkl")
joblib.dump(
    severity_model,
    "severity_model.pkl"
)

print("\nModels saved successfully!")