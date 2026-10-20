import joblib
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the training dataset
df = pd.read_csv("emails.csv")

# Display the first 5 emails
print(df.head())

# Step 3: Check how many emails in dataset
print("\nTotal emails:", len(df))

# Step 4: Count the phishing and legitimate emails
print("\nEmail classifications")
print(df["label"].value_counts())

# Separate email text and labels
x = df["text"]
y = df["label"]

# Create machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(x, y)

print("\nAI model trained successfully")

model_path = Path(__file__).with_name("phishing_model.joblib")
joblib.dump(model, model_path)

print("Model saved successfully!")

# Test model with new emails
test_emails = [
    "Your bank account will be suspended unless you verify your details immediately",
    "The project meeting is scheduled for tomorrow morning",
    "Urgent reminder to submit your assignment tomorrow"
]

for email in test_emails:
    prediction = model.predict([email])[0]
    phishing_score = model.predict_proba([email])[0][1]

    print("\nEmail:", email)
    print("Prediction", "Phishing" if prediction == 1 else "legitimate")
    print(f"Phishing score: {phishing_score:.2%}")

# New examples with expected labels
evaluation_emails = [
    "Your account will be disabled unless you confirm your login credentials",
    "Click here to update your bank password to prevent account closure",
    "You have won a cash prize submit your details to claim it",
    "Please find the meeting agenda for tomorrow",
    "Your assignment feedback is now available",
    "Reminder the project submission deadline is Friday"
]

# Expected classifications
# 1 phishing, 0 legitimate
actual_labels = [1, 1, 1, 0, 0, 0]

# Use the trained model to predict
predicted_labels = model.predict(evaluation_emails)

# Compare predictions with expected results
for email, actual, predicted in zip(
    evaluation_emails, actual_labels, predicted_labels
):
    print("\nEmail:", email)
    print("Expected:", actual)
    print("Predicted", predicted)
    print("Correct:", actual == predicted)

# Calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)

print(f'\nTest Accuracy: {accuracy:.2%}')

# Generate confusion matrix
cm = confusion_matrix(
    actual_labels, predicted_labels, labels=[0, 1]
)

print("\nConfusion Matrix:", cm)













