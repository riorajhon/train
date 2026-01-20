import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('training_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nNaN values per column:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

# Check for NaN values in variation column
nan_variations = df[df['variation'].isnull()]
print(f"\nRows with NaN variations: {len(nan_variations)}")

if len(nan_variations) > 0:
    print("Sample of problematic rows:")
    print(nan_variations.head())

# Check for any non-string values in variation column
print("\nChecking variation column data types:")
for i, val in enumerate(df['variation'].head(20)):
    print(f"Row {i}: {type(val)} - {repr(val)}")