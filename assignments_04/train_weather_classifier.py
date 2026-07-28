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

#Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",LogisticRegression(max_iter=1000,random_state=42))
])

param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train,y_train)

best_pipe = grid_search.best_estimator_
y_probs = best_pipe.predict_proba(X_test)[:,1]
y_pred = best_pipe.predict(X_test)
class_report = classification_report(y_test,y_pred)
test_auc = roc_auc_score(y_test, y_probs)

print(f"\nBest C: {grid_search.best_params_['clf__C']}\n")
print(f"\nBest CV AUC: {grid_search.best_score_:.3f}\n")
print(f"\nReport:\n {class_report}\n")
print(f"\nTest AUC: {test_auc}\n")

#Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test,y_probs)
fig, ax = plt.subplots(figsize=(6,5))

RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name=f"Logistic Regression (AUC={test_auc:.2f})")
ax.plot([0,1], linestyle="--",color="gray",label="Random")
ax.set_title("Weather Logistic Regression ROC")
plt.savefig("outputs/weather_roc.png")
plt.show()

#1. The model is "good" quality and AUC is suprisingly good.
#2. Looking at the precision & recall, false negatives are more common.
#3. Under-recommend it
#4. Threshold at >0.5 for better accuracy.
