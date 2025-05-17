from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Load dataset once at startup
DATA_PATH = "data/jee_data.csv"
df = pd.read_csv(DATA_PATH)

# Clean column names
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# FastAPI app
app = FastAPI(
    title="EduMentor AI - JEE College Predictor",
    description="Backend API for predicting college admissions based on JEE ranks",
    version="1.0.0"
)

# Allow frontend calls (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Schema
class PredictRequest(BaseModel):
    rank: int
    category: str
    gender: str
    quota: str

# Output Schema (optional for validation)
class CollegePrediction(BaseModel):
    institute: str
    program: str
    closing_rank: float
    tag: str

@app.post("/predict")
def predict(req: PredictRequest):
    # Filter dataset
    filtered = df[
        (df["seat_type"].str.upper() == req.category.upper()) &
        (df["gender"].str.lower() == req.gender.lower()) &
        (df["quota"].str.upper() == req.quota.upper())
    ]

    if filtered.empty:
        raise HTTPException(status_code=404, detail="No matching data found for the given input.")

    results = []

    for _, row in filtered.iterrows():
        avg = row["avg_closing_rank"]
        std = row["std_dev_closing_rank"]
        min_rank = row["min_closing_rank"]
        max_rank = row["max_closing_rank"]

        # Tag logic
        if req.rank <= min_rank:
            tag = "Safe"
        elif req.rank <= avg + std:
            tag = "Moderate"
        elif req.rank <= max_rank:
            tag = "Risky"
        else:
            tag = "Unlikely"

        results.append({
            "institute": row["institute"],
            "program": row["academic_program_name"],
            "closing_rank": round(avg, 1),
            "tag": tag
        })

    # Sort by closing rank
    results = sorted(results, key=lambda x: x["closing_rank"])
    return {"count": len(results), "results": results}
