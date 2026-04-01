from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import joblib

app = FastAPI(title="Cement Strength API", version="2.0")

pipeline = joblib.load("cement_pipeline.pkl")
FEATURES  = joblib.load("features.pkl")

class ConcreteInput(BaseModel):
    cement:           float = Field(..., gt=0)
    slag:             float = Field(0.0, ge=0)
    fly_ash:          float = Field(0.0, ge=0)
    water:            float = Field(..., gt=0)
    superplasticizer: float = Field(0.0, ge=0)
    coarse_agg:       float = Field(..., gt=0)
    fine_agg:         float = Field(..., gt=0)
    age:              int   = Field(..., gt=0, le=365)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "cement": 300.0,
                    "slag": 0.0,
                    "fly_ash": 0.0,
                    "water": 180.0,
                    "superplasticizer": 5.0,
                    "coarse_agg": 1000.0,
                    "fine_agg": 800.0,
                    "age": 28
                }
            ]
        }
    }

class PredictionOut(BaseModel):
    predicted_strength_mpa: float
    water_cement_ratio: float
    confidence_note: str

@app.get("/")
def health():
    return {"status": "ok", "model": "cement_pipeline.pkl"}

@app.post("/predict", response_model=PredictionOut)
def predict(data: ConcreteInput):  # <--- Match it to your class name above
    wc_ratio = data.water / data.cement
    if wc_ratio > 1.0:
        raise HTTPException(400, "Water/cement ratio > 1.0 is physically unrealistic")

    row = {
        'cement':           data.cement,
        'slag':             data.slag,
        'fly_ash':          data.fly_ash,
        'water':            data.water,
        'superplasticizer': data.superplasticizer,
        'coarse_agg':       data.coarse_agg,
        'fine_agg':         data.fine_agg,
        'log_age':          np.log1p(data.age),
        'water_cement_ratio': wc_ratio,
        'binder_total':     data.cement + data.slag + data.fly_ash,
    }

    X = pd.DataFrame([row])[FEATURES]
    pred = float(pipeline.predict(X)[0])

    note = "normal range" if 10 < pred < 80 else "check inputs — prediction outside typical range"

    return PredictionOut(
        predicted_strength_mpa=round(pred, 2),
        water_cement_ratio=round(wc_ratio, 3),
        confidence_note=note,
    )