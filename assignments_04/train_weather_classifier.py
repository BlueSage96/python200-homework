import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.99713980841658,
    "longitude": -78.33071903597848,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}
response = requests.get(url, params=params)
response.raise_for_status()
df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

def label_running_day(row):
    return int(
        7 <= row["temperature_2m_max"] <= 26
        and row["temperature_2m_min"] >= 0
        and row["precipitation_sum"] < 3.0
        and row["wind_speed_10m_max"] < 30
    )

df["good_for_running"] = df.apply(label_running_day, axis=1)
print(df)
print("Good for running:", df["good_for_running"].sum())

#The days labeled "good for running" is 148 days or 41%.
#I would say the percentage should be higher for my area, 
# but I don't know where to find the feature data for my town.

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",LogisticRegression(max_iter=1000,random_state=42))
])
