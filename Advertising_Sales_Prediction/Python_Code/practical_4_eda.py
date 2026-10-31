# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 4 — Exploratory Data Analysis (EDA)
# ============================================================
#  Unit Covered : Unit-1 & Unit-2
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('../Dataset/dataset.csv')
os.makedirs('../Reports', exist_ok=True)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("=" * 60)
print("PRACTICAL 4 : Exploratory Data Analysis (EDA)")
print("=" * 60)

# ── 1. Histograms ─────────────────────────────────────────
print("\n[1] Histograms...")
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle('Histograms – Numerical Features', fontsize=16, fontweight='bold')
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='black')
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig('../Reports/histogram.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: Reports/histogram.png")

# ── 2. Box Plots ──────────────────────────────────────────
print("\n[2] Box Plots...")
fig, axes = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle('Box Plots – Outlier Detection', fontsize=16, fontweight='bold')
axes = axes.flatten()
for i, col in enumerate(numerical_cols):
    axes[i].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    axes[i].set_title(col, fontsize=9)
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig('../Reports/boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: Reports/boxplot.png")

# ── 3. Pie Charts ─────────────────────────────────────────
print("\n[3] Pie Charts...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
df['Industry'].value_counts().plot.pie(
    ax=axes[0], autopct='%1.1f%%', startangle=140,
    colors=sns.color_palette('Set3', 10))
axes[0].set_title('Industry Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('')

df['Platform'].value_counts().plot.pie(
    ax=axes[1], autopct='%1.1f%%', startangle=90,
    colors=sns.color_palette('pastel', 6))
axes[1].set_title('Platform Distribution', fontsize=13, fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('../Reports/pie_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: Reports/pie_chart.png")

# ── 4. Bar Charts ─────────────────────────────────────────
print("\n[4] Bar Charts...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
df.groupby('Platform')['Sales_Lakhs'].mean().sort_values(ascending=False).plot(
    kind='bar', ax=axes[0], color='coral', edgecolor='black')
axes[0].set_title('Avg Sales by Platform', fontweight='bold')
axes[0].set_xlabel('Platform')
axes[0].set_ylabel('Avg Sales (₹ Lakhs)')
axes[0].tick_params(axis='x', rotation=30)

df.groupby('Season')['Sales_Lakhs'].mean().sort_values(ascending=False).plot(
    kind='bar', ax=axes[1], color='mediumseagreen', edgecolor='black')
axes[1].set_title('Avg Sales by Season', fontweight='bold')
axes[1].set_xlabel('Season')
axes[1].set_ylabel('Avg Sales (₹ Lakhs)')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig('../Reports/bar_chart.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: Reports/bar_chart.png")

# ── 5. Correlation Matrix ─────────────────────────────────
print("\n[5] Correlation Matrix...")
plt.figure(figsize=(13, 9))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../Reports/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Saved: Reports/correlation_matrix.png")

print("\n[✓] EDA complete. All charts saved in Reports/")
