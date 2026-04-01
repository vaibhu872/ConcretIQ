import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load the real UCI dataset (1030 samples, 8 numeric features) ──────────────
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
df = pd.read_excel(url)

# Clean column names
df.columns = [
    'cement', 'slag', 'fly_ash', 'water',
    'superplasticizer', 'coarse_agg', 'fine_agg', 'age', 'strength'
]

print(df.shape)        # (1030, 9)
print(df.describe())
print(df.isna().sum()) # Should be 0

# ── Feature engineering ───────────────────────────────────────────────────────
df['water_cement_ratio'] = df['water'] / df['cement']       # most important ratio
df['binder_total']       = df['cement'] + df['slag'] + df['fly_ash']
df['agg_total']          = df['coarse_agg'] + df['fine_agg']
df['log_age']            = np.log1p(df['age'])              # age effect is log-linear

# ── EDA plots ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0,0])
axes[0,0].set_title('Feature Correlations')

# Strength distribution
sns.histplot(df['strength'], kde=True, ax=axes[0,1])
axes[0,1].set_title('Strength Distribution')

# Water-cement ratio vs strength
axes[1,0].scatter(df['water_cement_ratio'], df['strength'], alpha=0.4)
axes[1,0].set_xlabel('Water/Cement Ratio')
axes[1,0].set_ylabel('Strength (MPa)')
axes[1,0].set_title('W/C Ratio vs Strength')

# Age vs strength
axes[1,1].scatter(df['log_age'], df['strength'], alpha=0.4, color='teal')
axes[1,1].set_xlabel('log(Age)')
axes[1,1].set_ylabel('Strength (MPa)')
axes[1,1].set_title('Age (log) vs Strength')

plt.tight_layout()
plt.savefig('eda.png', dpi=150)
plt.show()

df.to_csv('cement_clean.csv', index=False)
print("Saved cement_clean.csv")