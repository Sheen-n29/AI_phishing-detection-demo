
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

# STEP 1: Set project directory
project_dir = Path(__file__).resolve().parent

# STEP 2: Load Version 2 model
model_path = project_dir / "phishing_model_v2.joblib"

model = joblib.load(model_path)

print("Version 2 model loaded successfully!")

# STEP 3: Load held-out testing dataset
test_path = (
    project_dir / "data" / "processed" / "holdout_emails.csv"
)

test_data = pd.read_csv(test_path)

print("Total test emails:", len(test_data))

# STEP 4: Prepare test inputs and labels
X_test = test_data["text"]
y_test = test_data["label"]

# STEP 5: Generate predictions
predictions = model.predict(X_test)

# STEP 6: Calculate evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("\n=== VERSION 2 EVALUATION ===")

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1-score:  {f1:.2%}")

# STEP 7: Generate confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

# STEP 8: Generate classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Phishing"],
        labels=[0, 1],
        zero_division=0
    )
)

# STEP 9: Identify incorrect predictions
results = test_data.copy()

results["predicted"] = predictions

results["phishing_score"] = model.predict_proba(
    X_test
)[:, 1]

incorrect = results[
    results["label"] != results["predicted"]
]

print("\nTotal incorrect predictions:", len(incorrect))

print("\nSample misclassifications:")

print(
    incorrect[
        ["text", "label", "predicted", "phishing_score"]
    ].head(10).to_string(index=False)
)

# STEP 10: Save evaluation results
output_path = project_dir / "data" / "processed" / "v2_results.csv"

results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)
