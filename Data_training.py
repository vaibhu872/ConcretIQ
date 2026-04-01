import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    print("xgboost not installed — skipping. Run: pip install xgboost")
    USE_XGB = False

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('cement_clean.csv')

FEATURES = [
    'cement', 'slag', 'fly_ash', 'water',
    'superplasticizer', 'coarse_agg', 'fine_agg',
    'log_age', 'water_cement_ratio', 'binder_total'
]
TARGET = 'strength'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Define models (all wrapped in Pipeline with scaler) ───────────────────────
models = {
    'LinearRegression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    'Ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    'RandomForest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ]),
    'GradientBoosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(n_estimators=200, random_state=42))
    ]),
    'SVR': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
    ]),
}

if USE_XGB:
    models['XGBoost'] = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(n_estimators=200, learning_rate=0.05,
                               max_depth=6, random_state=42,
                               eval_metric='rmse', verbosity=0))
    ])

# ── Cross-validation comparison ───────────────────────────────────────────────
print(f"\n{'Model':<22} {'CV RMSE':>10} {'CV R²':>8}")
print("-" * 44)

results = {}
for name, pipe in models.items():
    cv_rmse = -cross_val_score(pipe, X_train, y_train,
                                cv=5, scoring='neg_root_mean_squared_error')
    cv_r2   =  cross_val_score(pipe, X_train, y_train,
                                cv=5, scoring='r2')
    results[name] = {'rmse': cv_rmse.mean(), 'r2': cv_r2.mean()}
    print(f"{name:<22} {cv_rmse.mean():>10.3f} {cv_r2.mean():>8.3f}")

# ── Pick best model & tune it ─────────────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]['rmse'])
print(f"\nBest model: {best_name}")

# Fine-tune the best model (example for RandomForest / XGBoost)
if best_name == 'RandomForest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth':    [None, 10, 20],
        'model__min_samples_split': [2, 5],
    }
elif best_name == 'XGBoost':
    param_grid = {
        'model__n_estimators':  [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth':     [4, 6, 8],
    }
else:
    param_grid = {}

best_pipe = models[best_name]

if param_grid:
    print("Running GridSearchCV (this takes ~1-2 min)...")
    gs = GridSearchCV(best_pipe, param_grid, cv=5,
                      scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_pipe = gs.best_estimator_
    print(f"Best params: {gs.best_params_}")
else:
    best_pipe.fit(X_train, y_train)

# ── Final evaluation on held-out test set ────────────────────────────────────
y_pred = best_pipe.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"\n── Test set results ({best_name}) ──")
print(f"  MAE  : {mae:.3f} MPa")
print(f"  RMSE : {rmse:.3f} MPa")
print(f"  R²   : {r2:.4f}")

# ── Feature importance (if tree model) ───────────────────────────────────────
import matplotlib.pyplot as plt

inner_model = best_pipe.named_steps['model']
if hasattr(inner_model, 'feature_importances_'):
    importances = pd.Series(inner_model.feature_importances_, index=FEATURES)
    importances.sort_values().plot(kind='barh', figsize=(8, 5), color='steelblue')
    plt.title(f'Feature Importance — {best_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()

# ── Save the best pipeline ────────────────────────────────────────────────────
joblib.dump(best_pipe, 'cement_pipeline.pkl')
joblib.dump(FEATURES,  'features.pkl')
print("\nSaved: cement_pipeline.pkl, features.pkl")