from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from noshow_iq.model import load_model, predict
from noshow_iq.preprocess import get_features_and_target  # noqa: F401

load_dotenv()

app = FastAPI()
client = MongoClient(os.getenv("MONGO_URI"))
db = client["noshowiq"]
predictions_col = db["predictions"]
training_runs_col = db["training_runs"]

model = load_model()


class AppointmentInput(BaseModel):
    age: int
    scholarship: int
    hypertension: int
    diabetes: int
    alcoholism: int
    handicap: int
    sms_received: int
    days_in_advance: int
    appt_day_of_week: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(data: AppointmentInput):
    input_dict = data.dict()
    X = pd.DataFrame([input_dict])

    risk, prob, recommendation = predict(model, X)

    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_input": input_dict,
        "cleaned_features": input_dict,
        "risk_level": risk,
        "probability": prob,
        "recommendation": recommendation,
    }
    predictions_col.insert_one(doc)

    return {
        "risk_level": risk,
        "probability": prob,
        "recommendation": recommendation,
    }


@app.get("/history")
def history():
    docs = list(predictions_col.find().sort("timestamp", -1).limit(20))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


@app.get("/stats")
def stats():
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk_level", "high"]}, 1, 0]}
                },
                "low_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk_level", "low"]}, 1, 0]}
                },
                "average_probability": {"$avg": "$probability"},
            }
        }
    ]

    result = list(predictions_col.aggregate(pipeline))

    last_run = training_runs_col.find_one(sort=[("timestamp", -1)])
    last_trained = last_run["timestamp"] if last_run else None

    if result:
        r = result[0]
        return {
            "total_predictions": r["total_predictions"],
            "high_risk_count": r["high_risk_count"],
            "low_risk_count": r["low_risk_count"],
            "average_probability": round(r["average_probability"], 4),
            "last_trained": last_trained,
        }
    return {
        "total_predictions": 0,
        "high_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": 0,
        "last_trained": last_trained,
    }
