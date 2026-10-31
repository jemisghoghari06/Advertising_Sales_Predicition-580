# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 2 — Dataset Collection & Description
# ============================================================
#  Unit Covered : Unit-1
# ============================================================

import pandas as pd

df = pd.read_csv('../Dataset/dataset.csv')

print("=" * 60)
print("PRACTICAL 2 : Dataset Collection & Description")
print("=" * 60)

print(f"\nShape  : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 Records:\n{df.head()}")

print("\n" + "=" * 60)
print("COLUMN DESCRIPTIONS")
print("=" * 60)
info = {
    'City'                  : 'Indian city of the campaign         [Categorical]',
    'Industry'              : 'Business sector of advertiser       [Categorical]',
    'Platform'              : 'Advertising channel used            [Categorical]',
    'Season'                : 'Season during campaign              [Categorical]',
    'TV_Spend_Lakhs'        : 'TV ad budget in ₹ Lakhs             [Numerical]',
    'Digital_Spend_Lakhs'   : 'Digital ad budget in ₹ Lakhs        [Numerical]',
    'Print_Spend_Lakhs'     : 'Print ad budget in ₹ Lakhs          [Numerical]',
    'Radio_Spend_Lakhs'     : 'Radio ad budget in ₹ Lakhs          [Numerical]',
    'Outdoor_Spend_Lakhs'   : 'Outdoor/OOH budget in ₹ Lakhs       [Numerical]',
    'Discount_Percent'      : 'Discount offered during campaign (%) [Numerical]',
    'Competitor_Spend_Lakhs': 'Competitor ad spend in ₹ Lakhs      [Numerical]',
    'Brand_Age_Years'       : 'Brand age in years                  [Numerical]',
    'Num_Outlets'           : 'Number of retail/distribution outlets[Numerical]',
    'Sales_Lakhs'           : '★ TARGET – Sales generated ₹ Lakhs  [Numerical]',
}
for col, desc in info.items():
    print(f"  {col:<28}: {desc}")

print("\nTarget Variable   : Sales_Lakhs")
print("Feature Variables :", [c for c in df.columns if c != 'Sales_Lakhs'])
