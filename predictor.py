import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "crop_production_model.pkl")
metadata = joblib.load(BASE_DIR / "models" / "model_metadata.pkl")


def predict_production(
    state_name,
    district_name,
    season,
    crop,
    crop_year,
    temperature,
    humidity,
    soil_moisture,
    area,
):
    # ---------- Feature Engineering ----------

    year_norm = (
        (crop_year - metadata["year_min"])
        / (metadata["year_max"] - metadata["year_min"])
    )

    log_area = np.log1p(area)

    temp_x_humidity = temperature * humidity

    temp_x_soil = temperature * soil_moisture

    season_code = metadata["season_map"].get(season, 0)

    # ---------- Create DataFrame ----------

    df = pd.DataFrame(
        {
            "State_Name": [state_name],
            "District_Name": [district_name],
            "Season": [season],
            "Crop": [crop],
            "Crop_Year": [crop_year],
            "Year_Norm": [year_norm],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Soil_Moisture": [soil_moisture],
            "log_Area": [log_area],
            "Temp_x_Humidity": [temp_x_humidity],
            "Temp_x_SoilMoisture": [temp_x_soil],
            "Season_Code": [season_code],
        }
    )

    prediction = model.predict(df)[0]

    return {
        "production": round(float(prediction), 2),
        "unit": "tonnes",
    }

if __name__ == "__main__":

    result = predict_production(
        state_name="West Bengal",
        district_name="NADIA",
        season="Kharif",
        crop="Rice",
        crop_year=2024,
        temperature=30,
        humidity=70,
        soil_moisture=55,
        area=100,
    )

    print(result)