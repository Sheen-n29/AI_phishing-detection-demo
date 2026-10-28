
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# STEP 1: Set project directory
project_dir = Path(__file__).resolve().parent

# STEP 2: Load the cleaned training dataset
train_path = (
    project_dir / "data" / "processed" / "train_emails.csv"
)

df = pd.read_csv(train_path)

print("Training dataset loaded successfully!")
print("Total training emails:", len(df))

print("\nClass distribution:")
print(df["label"].value_counts())

# STEP 3: Separate input features and labels
X_train = df["text"]
y_train = df["label"]

# STEP 4: Create the ML pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# STEP 5: Train the improved model
print("\nTraining Version 2 model...")

model.fit(X_train, y_train)

print("Model training completed!")

# STEP 6: Display vocabulary size
feature_count = len(
    model.named_steps["tfidf"].get_feature_names_out()
)

print("Total TF-IDF features:", feature_count)

# STEP 7: Save model as a separate version
model_path = project_dir / "phishing_model_v2.joblib"

joblib.dump(model, model_path)

print("\nVersion 2 model saved successfully!")
print("Saved model:", model_path.name)
