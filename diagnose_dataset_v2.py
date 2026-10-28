
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

# STEP 1: Locate the processed datasets
project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data" / "processed"

# STEP 2: Check both training and testing datasets
for filename in ["train_emails.csv", "holdout_emails.csv"]:

    df = pd.read_csv(data_dir / filename)

    print(f"\n=== {filename} ===")

    # STEP 3: Search for the signature "Best,"
    has_best = df["text"].str.contains(
        r"\bBest\s*,",
        case=False,
        regex=True,
        na=False
    )

    # STEP 4: Count occurrences in each class
    print("\nOccurrence of 'Best,' by label:")
    print(pd.crosstab(df["label"], has_best))

    # STEP 5: Classify using only the signature
    # Contains Best, => Legitimate (0)
    # Otherwise => Phishing (1)
    rule_predictions = (~has_best).astype(int)

    # STEP 6: Evaluate the simple rule
    accuracy = accuracy_score(
        df["label"], rule_predictions
    )

    print(f"\nSimple rule accuracy: {accuracy:.2%}")

    print("\nConfusion matrix:")
    print(confusion_matrix(
        df["label"],
        rule_predictions,
        labels=[0, 1]
    ))
