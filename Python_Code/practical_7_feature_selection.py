# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 7 — Feature Selection & Train-Test Split
# ============================================================
#  Unit Covered : Unit-3
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../Dataset/dataset_clean.csv')

print("=" * 60)
print("PRACTICAL 7 : Feature Selection & Train-Test Split")
print("=" * 60)

# ── Encode Categorical Columns ────────────────────────────
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
le = LabelEncoder()
df_enc = df.copy()
for col in categorical_cols:
    df_enc[col] = le.fit_transform(df_enc[col])
    print(f"  Encoded: {col}")

# ── Define Target & Features ──────────────────────────────
TARGET   = 'Sales_Lakhs'
FEATURES = [c for c in df_enc.columns if c != TARGET]

print(f"\nTarget Variable   : {TARGET}")
print(f"Feature Variables : {FEATURES}")

X = df_enc[FEATURES]
y = df_enc[TARGET]

# ── Correlation with Target ───────────────────────────────
print("\nCorrelation of each feature with Sales_Lakhs:")
corr_with_target = df_enc.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print(corr_with_target.round(4).to_string())

# Bar chart of correlations
plt.figure(figsize=(10, 6))
corr_with_target.plot(kind='barh', color=['green' if v > 0 else 'red' for v in corr_with_target])
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Correlation with Sales_Lakhs', fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.savefig('../Reports/feature_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Reports/feature_correlation.png")

# ── Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain-Test Split  : 80% Train / 20% Test")
print(f"Training Records  : {X_train.shape[0]}")
print(f"Testing Records   : {X_test.shape[0]}")
print(f"Feature Count     : {X_train.shape[1]}")

# Save processed data for next practical
import pickle, os
os.makedirs('../Dataset', exist_ok=True)
with open('../Dataset/split_data.pkl', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test, FEATURES, TARGET), f)
print("\n[✓] Split data saved → Dataset/split_data.pkl")
