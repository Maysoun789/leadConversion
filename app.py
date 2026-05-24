import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict


app = FastAPI(title="Lead Conversion Prediction API")

bundle = joblib.load("model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
tfidf = bundle["tfidf"]
decision_columns = bundle["decision_columns"]
positive_keywords = bundle["positive_keywords"]
negative_keywords = bundle["negative_keywords"]


class LeadInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    demande_devis: str = Field(alias="Demande Devis")
    connait_marque: str = Field(alias="Connait Marque")
    interest_level: str = Field(alias="Interest Level")
    decision_level: str = Field(alias="Decision Level")
    meeting_notes: str = Field(alias="Meeting Notes")


def bool_to_int(value):
    value = str(value).strip().upper()
    if value in ["TRUE", "1", "YES", "OUI"]:
        return 1
    return 0


def prepare_features(data: LeadInput):
    df = pd.DataFrame([{
        "Demande Devis": data.demande_devis,
        "Connait Marque": data.connait_marque,
        "Interest Level": data.interest_level,
        "Decision Level": data.decision_level,
        "Meeting Notes": data.meeting_notes
    }])

    df["Demande_Devis_enc"] = df["Demande Devis"].apply(bool_to_int)
    df["Connait_Marque_enc"] = df["Connait Marque"].apply(bool_to_int)

    interest_map = {
        "Froid": 0,
        "Tiède": 1,
        "Chaud": 2
    }
    df["Interest_Level_enc"] = df["Interest Level"].map(interest_map).fillna(0)

    decision_dummies = pd.get_dummies(df["Decision Level"], prefix="Decision", drop_first=False)
    decision_dummies = decision_dummies.reindex(columns=decision_columns, fill_value=0)

    tfidf_matrix = tfidf.transform(df["Meeting Notes"].fillna(""))
    x_tfidf = tfidf_matrix.toarray().astype(np.float64)

    df["notes_length"] = df["Meeting Notes"].fillna("").apply(len)
    df["notes_word_count"] = df["Meeting Notes"].fillna("").apply(lambda x: len(x.split()))

    for kw in positive_keywords:
        df[f"kw_{kw}"] = df["Meeting Notes"].fillna("").str.lower().str.contains(kw, na=False).astype(int)

    for kw in negative_keywords:
        df[f"kw_neg_{kw.replace(' ', '_')}"] = df["Meeting Notes"].fillna("").str.lower().str.contains(kw, na=False).astype(int)

    df["positive_kw_count"] = df[[f"kw_{kw}" for kw in positive_keywords]].sum(axis=1)
    df["negative_kw_count"] = df[[f"kw_neg_{kw.replace(' ', '_')}" for kw in negative_keywords]].sum(axis=1)

    numeric_features = [
        "Demande_Devis_enc",
        "Connait_Marque_enc",
        "Interest_Level_enc",
        "notes_length",
        "notes_word_count",
        "positive_kw_count",
        "negative_kw_count"
    ]

    keyword_features = [c for c in df.columns if c.startswith("kw_")]

    x_numeric = df[numeric_features + keyword_features].values.astype(np.float64)
    x_decision = decision_dummies.values.astype(np.float64)

    x_combined = np.hstack([x_numeric, x_decision, x_tfidf])
    x_combined = np.nan_to_num(x_combined, nan=0.0, posinf=0.0, neginf=0.0)

    x_scaled = scaler.transform(x_combined)
    x_scaled = np.nan_to_num(x_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return x_scaled


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Lead Conversion Prediction API is running",
        "model": bundle["best_model_name"]
    }


@app.post("/predict")
def predict(data: LeadInput):
    x = prepare_features(data)

    prediction = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1])

    return {
        "prediction": prediction,
        "converted": bool(prediction),
        "conversion_probability": round(probability, 4),
        "model": bundle["best_model_name"]
    }