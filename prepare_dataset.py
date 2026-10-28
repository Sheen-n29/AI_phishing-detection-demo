import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# Step 1: Set up file paths
project_dir = Path(__file__).resolve().parent

input_path = (
    project_dir / "data" /
    "phishing_legit_dataset_KD_10000.csv"
)

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 2: Load the dataset
df = pd.read_csv(input_path)

print("Original dataset:", df.shape)
print("\nOriginal class distribution:")
print(df["label"].value_counts())

# Step 3: Clean email text
def clean_email(text):
    text = str(text)

    # Remove artificial keywords lines
    text = re.sub(
        r"(?im)^[ \t]*Keywords[ \t]*:[^\n]*(?:\n|$)",
        " ",
        text
    )

    # Remove Subject marker but preserve subject text
    text = re.sub(
        r"(?im)^[ \t]*Subject[ \t]*:[ \t]*",
        "",
        text
    )

    # Remove unnecessary whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["clean_text"] = df["text"].apply(clean_email)

# STEP 4: Remove empty emails
df = df[df["clean_text"].str.len() > 0].copy()

# STEP 5: Check for conflicting labels
conflicts = (
    df.groupby("clean_text")["label"]
    .nunique()
    .gt(1)
    .sum()
)

if conflicts > 0:
    raise ValueError(
        f"Found {conflicts} texts with conflicting labels."
    )

# STEP 6: Remove duplicate cleaned emails
before = len(df)

df = df.drop_duplicates(subset=["clean_text"]).copy()

print("\nDuplicates removed:", before - len(df))

# STEP 7: Keep only the input and target
df["text"] = df["clean_text"]
df = df[["text", "label"]]

# STEP 8: Split into training and testing sets
train_data, test_data = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# STEP 9: Save the processed datasets
train_data.to_csv(
    output_dir / "train_emails.csv",
    index=False
)

test_data.to_csv(
    output_dir / "holdout_emails.csv",
    index=False
)

# STEP 10: Display the results
print("\nCleaned dataset:", df.shape)

print("\nTraining dataset:", train_data.shape)
print("Testing dataset:", test_data.shape)

print("\nTraining class distribution:")
print(train_data["label"].value_counts())

print("\nTesting class distribution:")
print(test_data["label"].value_counts())

print("\nDataset preparation completed!")
