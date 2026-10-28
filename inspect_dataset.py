
import pandas as pd
from pathlib import Path

# Locate the downloaded CSV files
data_folder = Path(__file__).resolve().parent / "data"
csv_files = sorted(data_folder.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in data folder.")

print("CSV files found:")
for file in csv_files:
    print("-", file.name)

# Use the first CSV for inspection
# If there are multiple files, choose the intended one
file_path = csv_files[0]

df = pd.read_csv(file_path)

print("\nDataset:", file_path.name)

# Number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# Column data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Inspect likely classification columns
for col in df.columns:
    name = col.lower()
    if any(term in name for term in ["label", "type", "class"]):
        print(f"\nValues in {col}:")
        print(df[col].value_counts(dropna=False).head(20))
