# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 5 — Handle Missing Values & Outliers
# ============================================================
#  Unit Covered : Unit-2
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../Dataset/dataset.csv')
numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("=" * 60)
print("PRACTICAL 5 : Handle Missing Values & Outliers")
print("=" * 60)

print(f"\n[BEFORE] Records     : {len(df)}")
print(f"[BEFORE] Missing     : {df.isnull().sum().sum()}")
print(f"[BEFORE] Duplicates  : {df.duplicated().sum()}")
print(f"[BEFORE] Sales range : {df['Sales_Lakhs'].min():.2f} – {df['Sales_Lakhs'].max():.2f}")

# ── Handle Missing Values ─────────────────────────────────
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Filled '{col}' with median")
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  Filled '{col}' with mode")
print(f"  Missing after treatment: {df.isnull().sum().sum()}")

# ── Remove Duplicates ─────────────────────────────────────
df.drop_duplicates(inplace=True)

# ── Handle Outliers (IQR method) ─────────────────────────
Q1  = df['Sales_Lakhs'].quantile(0.25)
Q3  = df['Sales_Lakhs'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"\n  IQR Bounds → Lower: {lower:.2f}  Upper: {upper:.2f}")
print(f"  Outliers found: {len(df[(df['Sales_Lakhs']<lower)|(df['Sales_Lakhs']>upper)])}")

df_clean = df[(df['Sales_Lakhs'] >= lower) & (df['Sales_Lakhs'] <= upper)].copy()

print(f"\n[AFTER]  Records     : {len(df_clean)}")
print(f"[AFTER]  Missing     : {df_clean.isnull().sum().sum()}")
print(f"[AFTER]  Sales range : {df_clean['Sales_Lakhs'].min():.2f} – {df_clean['Sales_Lakhs'].max():.2f}")

# ── Before vs After Plot ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Before vs After Outlier Removal – Sales_Lakhs', fontsize=13, fontweight='bold')
axes[0].boxplot(df['Sales_Lakhs'], patch_artist=True, boxprops=dict(facecolor='lightcoral'))
axes[0].set_title(f'BEFORE  (n={len(df)})')
axes[0].set_ylabel('Sales (₹ Lakhs)')
axes[1].boxplot(df_clean['Sales_Lakhs'], patch_artist=True, boxprops=dict(facecolor='lightgreen'))
axes[1].set_title(f'AFTER  (n={len(df_clean)})')
axes[1].set_ylabel('Sales (₹ Lakhs)')
plt.tight_layout()
plt.savefig('../Reports/before_after_outliers.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Saved: Reports/before_after_outliers.png")

df_clean.to_csv('../Dataset/dataset_clean.csv', index=False)
print("[✓] Cleaned dataset saved → Dataset/dataset_clean.csv")
