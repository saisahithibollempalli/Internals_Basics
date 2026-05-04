from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

app = FastAPI()

model = joblib.load("../models/model.pkl")

class InputData(BaseModel):
    slope_degrees: float = Field(ge=5, le=45)
    rainfall_mm: float = Field(ge=50, le=500)
    soil_depth_m: float = Field(ge=0.5, le=5)
    vegetation_index: float = Field(ge=0.1, le=0.9)

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(data: InputData):
    vals = [[
        data.slope_degrees,
        data.rainfall_mm,
        data.soil_depth_m,
        data.vegetation_index
    ]]
    pred = model.predict(vals)[0]
    return {"prediction": pred}