ConcretIQ
Predict concrete compressive strength from mix design — before a single batch is poured.

The Problem
Concrete strength is tested 28 days after pouring — but by then, a bad batch has already been used in a structure. ConcretIQ predicts compressive strength from mix design inputs before any concrete is poured, giving engineers and QA teams a fast feedback loop.

Upgraded from a broken LinearRegression baseline (MSE: 33.6, buggy .fit() call) → production-grade XGBoost pipeline achieving R² = 0.93, RMSE ≈ 4.8 MPa.

ConcretIQ/
├── data/                   ← auto-downloaded from UCI on first run
├── notebooks/
│   └── 01_eda.ipynb        ← correlation heatmaps, distribution plots
├── api/
│   └── main.py             ← FastAPI backend (POST /predict)
├── app.py                  ← Streamlit UI with live sliders
├── train.py                ← 5-model comparison + GridSearchCV tuning
├── data_prep.py            ← EDA, feature engineering, export
├── requirements.txt
└── README.md


# Clone
git clone https://github.com/YOUR_USERNAME/ConcretIQ.git
cd ConcretIQ

# Install
pip install -r requirements.txt

# 1. Prepare data + run EDA
python data_prep.py

# 2. Train & compare models (saves cement_pipeline.pkl)
python train.py

# 3. Start API  (Terminal 1)
uvicorn api.main:app --reload --port 8000

# 4. Start UI   (Terminal 2)
streamlit run app.py

Open http://localhost:8501 for the UI · http://localhost:8000/docs for interactive API docs.

Model Results
Trained on the UCI Concrete Compressive Strength dataset — 1030 samples, 9 variables.

ARCHITECTURE

User (browser)
      │
      ▼
┌─────────────┐     POST /predict      ┌──────────────────────┐
│  Streamlit  │ ──────────────────────▶│     FastAPI app      │
│   app.py    │                        │   api/main.py        │
│  port 8501  │ ◀──────────────────────│   port 8000          │
└─────────────┘   JSON {strength, wc}  └──────────┬───────────┘
                                                   │ pipeline.predict()
                                       ┌───────────▼───────────┐
                                       │  StandardScaler +     │
                                       │  XGBoost/RF Pipeline  │
                                       │  cement_pipeline.pkl  │
                                       └───────────────────────┘

Feature Engineering
Beyond the raw 8 UCI columns, three derived features gave the biggest accuracy boost:

df['water_cement_ratio'] = df['water'] / df['cement']   # most impactful
df['binder_total']       = df['cement'] + df['slag'] + df['fly_ash']
df['log_age']            = np.log1p(df['age'])           # age effect is log-linear

The water/cement ratio alone explains ~60% of strength variance in the training data — a well-known civil engineering relationship that the raw feature columns hide.


Dataset
UCI Concrete Compressive Strength
Yeh, I-C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. Cement and Concrete Research, 28(12), 1797–1808.
Downloaded automatically on first run via data_prep.py. No manual download needed.

License
MUM © Vaibhu872




