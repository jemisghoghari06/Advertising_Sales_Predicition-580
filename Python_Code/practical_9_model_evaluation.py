# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 9 — Model Evaluation
# ============================================================
#  Unit Covered : Unit-3 & Unit-4
# ============================================================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

with open('../Dataset/model_results.pkl', 'rb') as f:
    model, y_pred, y_test, X_train, X_test, FEATURES = pickle.load(f)

print("=" * 60)
print("PRACTICAL 9 : Model Evaluation (Regression Metrics)")
print("=" * 60)

# ── Regression Metrics ────────────────────────────────────
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y_test.values - y_pred))

print(f"\n  R² Score (R-Squared) : {r2:.4f}")
print(f"  MSE  (Mean Sq Error) : {mse:.4f}")
print(f"  RMSE (Root MSE)      : {rmse:.4f}")
print(f"  MAE  (Mean Abs Err)  : {mae:.4f}")

print(f"\n  Interpretation:")
print(f"  → R²={r2:.2f} means the model explains {r2*100:.1f}% of variance in Sales.")
print(f"  → RMSE={rmse:.2f}: avg prediction error is ₹{rmse:.2f} Lakhs.")

# ── Actual vs Predicted Plot ──────────────────────────────
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='royalblue', alpha=0.6, label='Predictions')
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Sales (₹ Lakhs)')
plt.ylabel('Predicted Sales (₹ Lakhs)')
plt.title('Actual vs Predicted Sales', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('../Reports/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Saved: Reports/actual_vs_predicted.png")

# ── Residual Plot ─────────────────────────────────────────
residuals = y_test.values - y_pred
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, color='darkorange', alpha=0.6)
plt.axhline(0, color='red', linestyle='--', lw=1.5)
plt.xlabel('Predicted Sales (₹ Lakhs)')
plt.ylabel('Residuals')
plt.title('Residual Plot', fontweight='bold')
plt.tight_layout()
plt.savefig('../Reports/residual_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Reports/residual_plot.png")

# ── Feature Importance (Coefficients) ────────────────────
coef_df = pd.DataFrame({'Feature': FEATURES, 'Coefficient': model.coef_})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)

plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
plt.axvline(0, color='black', lw=0.8)
plt.title('Feature Coefficients (Impact on Sales)', fontweight='bold')
plt.xlabel('Coefficient Value')
plt.tight_layout()
plt.savefig('../Reports/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Saved: Reports/feature_importance.png")

print("\n[✓] Model evaluation complete.")
