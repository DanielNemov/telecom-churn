import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'etl'))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from extract import extract
from transform import transform

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
sns.set_theme(style='whitegrid', palette='muted')

IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports', 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

raw_df = extract()
raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce')
raw_df.dropna(subset=['TotalCharges'], inplace=True)
processed_df = transform(raw_df.copy())

# 1. Churn distribution
counts = raw_df['Churn'].value_counts()
colors = ['#4C72B0', '#DD8452']
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.5)
axes[0].set_title('Churn Count', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')
axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=colors,
            startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Churn Rate', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'churn_distribution.png'), bbox_inches='tight')
plt.close()
print('1/5 churn_distribution.png')

# 2. Numeric features by churn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['tenure', 'MonthlyCharges', 'TotalCharges']):
    for label, color in zip(['No', 'Yes'], colors):
        ax.hist(raw_df[raw_df['Churn'] == label][col], bins=30,
                alpha=0.6, color=color, label=f'Churn={label}', edgecolor='none')
    ax.set_title(col, fontsize=13, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    ax.legend()
plt.suptitle('Numeric Features by Churn', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'numeric_features_by_churn.png'), bbox_inches='tight')
plt.close()
print('2/5 numeric_features_by_churn.png')

# 3. Feature correlation with Churn
churn_corr = processed_df.corr()['Churn'].drop('Churn').sort_values(key=abs, ascending=False)
bar_colors = ['#DD8452' if v > 0 else '#4C72B0' for v in churn_corr.values]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(churn_corr.index, churn_corr.values, color=bar_colors, edgecolor='white')
ax.set_xlabel('Pearson Correlation with Churn')
ax.set_title('Feature Correlation with Churn', fontsize=14, fontweight='bold')
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.legend(handles=[mpatches.Patch(color='#DD8452', label='Positive'),
                   mpatches.Patch(color='#4C72B0', label='Negative')])
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'feature_correlation_with_churn.png'), bbox_inches='tight')
plt.close()
print('3/5 feature_correlation_with_churn.png')

# 4. Correlation matrix
corr = processed_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax, annot_kws={'size': 8})
ax.set_title('Correlation Matrix', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'correlation_matrix.png'), bbox_inches='tight')
plt.close()
print('4/5 correlation_matrix.png')

# 5. Boxplots
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
sns.boxplot(data=raw_df, x='Contract', y='MonthlyCharges', hue='Churn',
            palette=['#4C72B0', '#DD8452'], ax=axes[0])
axes[0].set_title('Monthly Charges by Contract & Churn', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=10)
sns.boxplot(data=raw_df, x='Contract', y='tenure', hue='Churn',
            palette=['#4C72B0', '#DD8452'], ax=axes[1])
axes[1].set_title('Tenure by Contract & Churn', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=10)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'boxplots_contract.png'), bbox_inches='tight')
plt.close()
print('5/5 boxplots_contract.png')

print(f'\nAll images saved to {os.path.abspath(IMAGES_DIR)}')
