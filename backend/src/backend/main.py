from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

# Path to model
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "model.pkl"

# Load model once when app starts
model = joblib.load(MODEL_PATH)

app = FastAPI(title="ML Prediction API")

# TODO: Adjust fields based on your dataset
class InputData(BaseModel):
    feature1: float
    feature2: float
    # add all dataset features (numerical + categorical)

@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": str(prediction)}
