from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(title="NBA Fantasy Points Predictor")

model = joblib.load("models/xgb_fantasy_model.pkl")
features = joblib.load("models/features.pkl")


class PlayerFeatures(BaseModel):
    points_last_3: float
    points_last_5: float
    assists_last_3: float
    assists_last_5: float
    reboundsTotal_last_3: float
    reboundsTotal_last_5: float
    steals_last_3: float
    steals_last_5: float
    blocks_last_3: float
    blocks_last_5: float
    turnovers_last_3: float
    turnovers_last_5: float
    foulsPersonal_last_3: float
    foulsPersonal_last_5: float
    numMinutes: float
    numMinutes_missing: int
    rolling_minutes: float
    player_median_minutes: float
    home: int
    is_playoff: int
    fp_per_min_last_5: float


@app.get("/")
def root():
    return {"message": "NBA Fantasy Points API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: PlayerFeatures):

    row = data.model_dump()

    # recreate engineered feature 
    row["expected_fp"] = row["fp_per_min_last_5"] * row["rolling_minutes"]

    # Convert to DataFrame
    X = pd.DataFrame([row])

    # Ensure exact feature order
    X = X[features]

    prediction = model.predict(X)[0]

    return {
        "predicted_fantasy_points": round(float(prediction), 2)
    }