from fastapi import FastAPI
from pydantic import BaseModel

from predictor import predict_production

app = FastAPI(
    title="Crop Production Prediction API",
    version="1.0.0"
)


class PredictionRequest(BaseModel):
    state_name: str
    district_name: str
    season: str
    crop: str
    crop_year: int
    temperature: float
    humidity: float
    soil_moisture: float
    area: float


@app.get("/")
def home():
    return {
        "message": "Crop Prediction API is Running"
    }


@app.post("/predict")
def predict(data: PredictionRequest):

    prediction = predict_production(
        state_name=data.state_name,
        district_name=data.district_name,
        season=data.season,
        crop=data.crop,
        crop_year=data.crop_year,
        temperature=data.temperature,
        humidity=data.humidity,
        soil_moisture=data.soil_moisture,
        area=data.area,
    )

    return {
        "prediction": prediction
    }