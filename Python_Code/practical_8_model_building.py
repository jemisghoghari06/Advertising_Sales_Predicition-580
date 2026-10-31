# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 8 — Build ML Model (Linear Regression)
# ============================================================
#  Unit Covered : Unit-4
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

# ── Load prepared data ────────────────────────────────────
with open('../Dataset/split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, FEATURES, TARGET = pickle.load(f)

print("=" * 60)
print("PRACTICAL 8 : Model Building – Linear Regression")
print("=" * 60)
print(f"\nAlgorithm : Linear Regression")
print(f"Problem   : Regression (predicting continuous Sales value)")
print(f"Features  : {len(FEATURES)}")
print(f"Train set : {X_train.shape[0]} records")
print(f"Test set  : {X_test.shape[0]} records")

# ── Train the Model ───────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("\n[✓] Model trained successfully!")

# ── Coefficients ──────────────────────────────────────────
print("\nModel Intercept :", round(model.intercept_, 4))
coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': model.coef_})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
print("\nFeature Coefficients (sorted by importance):")
print(coef_df.to_string(index=False))

# ── Predictions ───────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nSample Predictions (first 10):")
compare = pd.DataFrame({'Actual': y_test.values[:10].round(2),
                        'Predicted': y_pred[:10].round(2)})
compare['Difference'] = (compare['Actual'] - compare['Predicted']).round(2)
print(compare.to_string(index=False))

# Save model and predictions for next practical
with open('../Dataset/model_results.pkl', 'wb') as f:
    pickle.dump((model, y_pred, y_test, X_train, X_test, FEATURES), f)
print("\n[✓] Model saved → Dataset/model_results.pkl")
