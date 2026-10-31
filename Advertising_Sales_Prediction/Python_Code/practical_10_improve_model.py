# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 10 — Improve Model (CV + Regularisation)
# ============================================================
#  Unit Covered : Unit-4
# ============================================================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

with open('../Dataset/split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, FEATURES, TARGET = pickle.load(f)

print("=" * 60)
print("PRACTICAL 10 : Improve Model – CV & Regularisation")
print("=" * 60)

X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

# ── 1. Cross Validation on Linear Regression ─────────────
lr = LinearRegression()
cv_scores = cross_val_score(lr, X_all, y_all, cv=5, scoring='r2')
print(f"\n[1] 5-Fold Cross Validation – Linear Regression")
for i, s in enumerate(cv_scores, 1):
    print(f"    Fold {i}: R² = {s:.4f}")
print(f"    Mean R²  : {cv_scores.mean():.4f}")
print(f"    Std Dev  : {cv_scores.std():.4f}")

# ── 2. Ridge Regression ───────────────────────────────────
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred  = ridge.predict(X_test)
ridge_r2    = r2_score(y_test, ridge_pred)
ridge_rmse  = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_cv    = cross_val_score(ridge, X_all, y_all, cv=5, scoring='r2').mean()
print(f"\n[2] Ridge Regression (alpha=1.0)")
print(f"    R²   : {ridge_r2:.4f}  |  RMSE: {ridge_rmse:.4f}  |  CV R²: {ridge_cv:.4f}")

# ── 3. Lasso Regression ───────────────────────────────────
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred  = lasso.predict(X_test)
lasso_r2    = r2_score(y_test, lasso_pred)
lasso_rmse  = np.sqrt(mean_squared_error(y_test, lasso_pred))
lasso_cv    = cross_val_score(lasso, X_all, y_all, cv=5, scoring='r2').mean()
print(f"\n[3] Lasso Regression (alpha=0.1)")
print(f"    R²   : {lasso_r2:.4f}  |  RMSE: {lasso_rmse:.4f}  |  CV R²: {lasso_cv:.4f}")

# ── 4. Comparison Table ───────────────────────────────────
lr.fit(X_train, y_train)
lr_pred  = lr.predict(X_test)
lr_r2    = r2_score(y_test, lr_pred)
lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_pred))

print(f"\n{'='*60}")
print(f"  MODEL COMPARISON TABLE")
print(f"  {'Model':<22} {'Test R²':>8} {'RMSE':>9} {'CV R²':>9}")
print(f"  {'-'*52}")
print(f"  {'Linear Regression':<22} {lr_r2:>8.4f} {lr_rmse:>9.4f} {cv_scores.mean():>9.4f}")
print(f"  {'Ridge Regression':<22} {ridge_r2:>8.4f} {ridge_rmse:>9.4f} {ridge_cv:>9.4f}")
print(f"  {'Lasso Regression':<22} {lasso_r2:>8.4f} {lasso_rmse:>9.4f} {lasso_cv:>9.4f}")
print(f"{'='*60}")

# ── 5. Comparison Bar Chart ───────────────────────────────
models  = ['Linear', 'Ridge', 'Lasso']
r2_vals = [lr_r2, ridge_r2, lasso_r2]
cv_vals = [cv_scores.mean(), ridge_cv, lasso_cv]

x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, r2_vals, width, label='Test R²',  color='steelblue',  edgecolor='black')
bars2 = ax.bar(x + width/2, cv_vals, width, label='CV R² (mean)', color='coral', edgecolor='black')
ax.set_title('Model Comparison – R² Scores', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('R² Score')
ax.set_ylim(0, 1)
ax.legend()
for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('../Reports/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Saved: Reports/model_comparison.png")
print("\n[✓] Model improvement complete.")
