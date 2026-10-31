# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 3 — Understanding Data
# ============================================================
#  Unit Covered : Unit-1
# ============================================================

import pandas as pd
import numpy as np

df = pd.read_csv('../Dataset/dataset.csv')

print("=" * 60)
print("PRACTICAL 3 : Understanding Data")
print("=" * 60)

print("\n1. DATA TYPES")
print(df.dtypes)

numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\n2. NUMERICAL COLUMNS ({len(numerical_cols)})\n   {numerical_cols}")
print(f"\n3. CATEGORICAL COLUMNS ({len(categorical_cols)})\n   {categorical_cols}")

print("\n4. UNIQUE VALUES IN CATEGORICAL COLUMNS")
for col in categorical_cols:
    print(f"   {col}: {sorted(df[col].unique().tolist())}")

print(f"\n5. MISSING VALUES\n{df.isnull().sum()}")
print(f"\n   Total missing: {df.isnull().sum().sum()}")

print(f"\n6. DUPLICATE ROWS: {df.duplicated().sum()}")

print("\n7. STATISTICAL SUMMARY")
print(df.describe().round(2))

print("\n" + "=" * 60)
print("SUMMARY")
print(f"  Records        : {df.shape[0]}")
print(f"  Columns        : {df.shape[1]}")
print(f"  Numerical Cols : {len(numerical_cols)}")
print(f"  Categorical    : {len(categorical_cols)}")
print(f"  Missing Values : {df.isnull().sum().sum()}")
print(f"  Duplicates     : {df.duplicated().sum()}")
print("=" * 60)
