from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
import pandas as pd
import joblib
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="KSI Accident Severity Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = "model_artifacts"

model = joblib.load(f"{ARTIFACTS_DIR}/ksi_xgboost_final.pkl")
imputer = joblib.load(f"{ARTIFACTS_DIR}/iterative_imputer.pkl")
target_encoder = joblib.load(f"{ARTIFACTS_DIR}/target_encoder.pkl")

with open(f"{ARTIFACTS_DIR}/threshold_config.json", "r", encoding="utf-8") as f:
    threshold_config = json.load(f)

THRESHOLD = threshold_config["threshold"]
FEATURE_COLUMNS = threshold_config["feature_columns"]

IMPUTE_COLUMNS = ["avg_driver_age", "max_engine_cc", "avg_vehicle_age", "veh_count", "n_male_drivers"]


class AccidentInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Location_Easting_OSGR: float
    Location_Northing_OSGR: float
    Longitude: float
    Latitude: float
    Police_Force: int
    Number_of_Casualties: int
    Date: str  # format: DD/MM/YYYY
    Day_of_Week: int  # 1 = Sunday ... 7 = Saturday
    Time: str  # format: HH:MM
    Local_Authority_District: int = Field(alias="Local_Authority_(District)")
    Local_Authority_Highway: str = Field(alias="Local_Authority_(Highway)")
    First_Road_Class: int = Field(alias="1st_Road_Class")
    First_Road_Number: int = Field(alias="1st_Road_Number")
    Road_Type: int
    Speed_limit: int
    Junction_Detail: int
    Junction_Control: int
    Second_Road_Class: int = Field(alias="2nd_Road_Class")
    Second_Road_Number: int = Field(alias="2nd_Road_Number")
    Pedestrian_Crossing_Human_Control: int = Field(alias="Pedestrian_Crossing-Human_Control")
    Pedestrian_Crossing_Physical_Facilities: int = Field(alias="Pedestrian_Crossing-Physical_Facilities")
    Light_Conditions: int
    Weather_Conditions: int
    Road_Surface_Conditions: int
    Special_Conditions_at_Site: int
    Carriageway_Hazards: int
    Urban_or_Rural_Area: int
    Did_Police_Officer_Attend_Scene_of_Accident: int
    veh_count: int
    avg_driver_age: Optional[float] = None
    max_engine_cc: Optional[float] = None
    avg_vehicle_age: Optional[float] = None
    n_male_drivers: int


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    threshold_used: float


def clean_raw_values(row: dict) -> dict:
    # same sanity-check rules applied during training (cell 12 in the notebook)
    if row.get("max_engine_cc") is not None and row["max_engine_cc"] > 10000:
        row["max_engine_cc"] = None
    if row.get("avg_driver_age") is not None and row["avg_driver_age"] < 15:
        row["avg_driver_age"] = None
    return row


def build_feature_row(payload: AccidentInput) -> pd.DataFrame:
    row = payload.model_dump(by_alias=True)
    row = clean_raw_values(row)

    df = pd.DataFrame([row])

    # missing-value indicators (must be computed BEFORE imputation, same order as training)
    df["driver_age_missing"] = df["avg_driver_age"].isna().astype(int)
    df["engine_cc_missing"] = df["max_engine_cc"].isna().astype(int)
    df["vehicle_age_missing"] = df["avg_vehicle_age"].isna().astype(int)

    # use the FITTED imputer, transform only (never fit again at inference time)
    df[IMPUTE_COLUMNS] = imputer.transform(df[IMPUTE_COLUMNS])

    try:
        date_obj = pd.to_datetime(df["Date"], dayfirst=True)
        time_obj = pd.to_datetime(df["Time"], format="%H:%M")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Date/Time format: {e}")

    df["Hour"] = time_obj.dt.hour
    df["Month"] = date_obj.dt.month
    df["Is_Weekend"] = df["Day_of_Week"].isin([1, 7]).astype(int)
    df = df.drop(columns=["Date", "Time"])

    # use the FITTED target encoder, transform only
    df["Local_Authority_(Highway)"] = target_encoder.transform(df["Local_Authority_(Highway)"])

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(status_code=500, detail=f"Missing engineered columns: {missing_cols}")

    # reorder to EXACTLY match the columns the model was trained on
    df = df[FEATURE_COLUMNS]
    return df


def classify_risk(probability: float) -> str:
    # example cutoffs, not derived from the data — adjust freely
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    return "High"


@app.get("/")
def root():
    return {"status": "ok", "model": "ksi_xgboost_final", "threshold": THRESHOLD}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: AccidentInput):
    X = build_feature_row(payload)
    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= THRESHOLD)

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        risk_level=classify_risk(probability),
        threshold_used=THRESHOLD,
    )

# uvicorn main:app --reload