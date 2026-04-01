# ConcretIQ

Predict concrete compressive strength from mix design inputs using
Random Forest / XGBoost — upgraded from a basic LinearRegression baseline.

## Features
- 5-model comparison with 5-fold cross-validation
- FastAPI REST backend (`POST /predict`)
- Streamlit UI with live sliders
- Feature engineering: water/cement ratio, log(age), binder total

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/ConcretIQ.git
cd ConcretIQ
pip install -r requirements.txt

# Train the model
python train.py

# Start API (terminal 1)
uvicorn api.main:app --reload --port 8000

# Start UI (terminal 2)
streamlit run app.py
```

## Dataset
UCI Concrete Compressive Strength — 1030 samples, 8 features.
Download automatically via `train.py`.

## Results

| Model | CV RMSE | R² |
|---|---|---|
| LinearRegression (baseline) | ~8.2 MPa | 0.78 |
| Random Forest | ~5.1 MPa | 0.91 |
| XGBoost | ~4.8 MPa | 0.93 |