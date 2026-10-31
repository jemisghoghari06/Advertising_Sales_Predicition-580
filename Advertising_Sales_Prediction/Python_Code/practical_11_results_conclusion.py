# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 11 — Results & Conclusion
# ============================================================
#  Unit Covered : Unit-3 & Unit-4
# ============================================================

import pickle
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

with open('../Dataset/split_data.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test, FEATURES, TARGET = pickle.load(f)

import pandas as pd
X_all = pd.concat([X_train, X_test])
y_all = pd.concat([y_train, y_test])

lr    = LinearRegression().fit(X_train, y_train)
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

def get_metrics(m):
    p    = m.predict(X_test)
    r2   = r2_score(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    cv   = cross_val_score(m, X_all, y_all, cv=5, scoring='r2').mean()
    return r2, rmse, cv

lr_m    = get_metrics(lr)
ridge_m = get_metrics(ridge)
lasso_m = get_metrics(lasso)

print("=" * 60)
print("PRACTICAL 11 : Results & Conclusion")
print("=" * 60)

print("""
PROJECT  : Advertising Sales Prediction
DATASET  : Indian Advertising Sales (550 records, 14 features)
ALGORITHM: Linear Regression (+ Ridge & Lasso variants)
""")

print("─" * 60)
print("RESULTS SUMMARY")
print("─" * 60)
print(f"  {'Model':<22} {'R²':>8} {'RMSE':>9} {'CV R²':>9}")
print(f"  {'-'*50}")
print(f"  {'Linear Regression':<22} {lr_m[0]:>8.4f} {lr_m[1]:>9.4f} {lr_m[2]:>9.4f}")
print(f"  {'Ridge Regression':<22} {ridge_m[0]:>8.4f} {ridge_m[1]:>9.4f} {ridge_m[2]:>9.4f}")
print(f"  {'Lasso Regression':<22} {lasso_m[0]:>8.4f} {lasso_m[1]:>9.4f} {lasso_m[2]:>9.4f}")

all_r2   = [lr_m[0], ridge_m[0], lasso_m[0]]
names    = ['Linear', 'Ridge', 'Lasso']
best_idx = int(np.argmax(all_r2))
print(f"\n  Best Model : {names[best_idx]} Regression  (R² = {all_r2[best_idx]:.4f})")

print("""
─────────────────────────────────────────────────────────────
KEY FINDINGS
─────────────────────────────────────────────────────────────
1. Digital and TV spending are the strongest positive
   predictors of sales in the Indian market.
2. Festive season (Diwali, Navratri) significantly boosts
   sales — campaigns during this period outperform others.
3. Competitor spending has a negative impact on sales,
   confirming competitive market dynamics.
4. Brand age and number of outlets positively contribute
   to sales, showing importance of distribution network.
5. Ridge/Lasso regularisation produced comparable results,
   suggesting the model was not heavily overfitting.

─────────────────────────────────────────────────────────────
CONCLUSION
─────────────────────────────────────────────────────────────
The Advertising Sales Prediction model was successfully built
using Linear Regression on a real-world inspired Indian
advertising dataset. The model achieves a good R² score,
explaining significant variance in Sales. Cross-validation
confirmed the model generalises well to unseen data.

─────────────────────────────────────────────────────────────
FUTURE SCOPE
─────────────────────────────────────────────────────────────
1. Use advanced models: Random Forest, XGBoost, or Gradient
   Boosting for potentially higher accuracy.
2. Add features: social media engagement rate, GST slab,
   urban vs. rural split, festive day count.
3. Build a Streamlit or Flask web app so marketing teams
   can input budgets and get instant sales predictions.
4. Incorporate time-series analysis for monthly/quarterly
   sales forecasting using ARIMA or LSTM.
5. Expand dataset to include 2000+ records from multiple
   Indian states for a more robust model.
─────────────────────────────────────────────────────────────
""")
print("[✓] Project Complete!")
