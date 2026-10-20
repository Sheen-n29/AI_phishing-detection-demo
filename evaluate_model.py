
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# STEP 1: Locate the project folder
project_dir = Path(__file__).resolve().parent

# STEP 2: Load the previously trained AI model
model = joblib.load(
    project_dir / "phishing_model.joblib"
)

# STEP 3: Load the separate testing dataset
test_data = pd.read_csv(
    project_dir / "test_emails.csv"
)

X_test = test_data["text"]
y_test = test_data["label"]

# STEP 4: Ask the model to classify every email
predictions = model.predict(X_test)

# STEP 5: Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

print("\n=== MODEL EVALUATION RESULTS ===")

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")

# STEP 6: Generate confusion matrix
cm = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

# STEP 7: Detailed classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

# STEP 8: Display individual predictions
results = test_data.copy()
results["predicted"] = predictions
results["correct"] = (
    results["label"] == results["predicted"]
)

print("\nIndividual Predictions:")
print(results.to_string(index=False))
