import joblib
import json 
import pandas as pd

#Load pipeline
weather_clf = joblib.load("models/weather_classifier.pkl")
print("Weather:", weather_clf)

#Load metadata
with open("models/weather_classifier_metadata.json",'r') as f:
    metadata_clf = json.load(f)
    
#Task 01

print(f"\nTask 01:\n")
print(f"City: {metadata_clf['city']}")
print(f"Features: {metadata_clf['features']}")
print(f"Test AUC: {metadata_clf['test_auc']}")

#Task 02
print(f"\nTask 02:\n")
#data frame for made up data
new_days = pd.DataFrame({
    "temperature_2m_max": [32.0, 15.0, 7.0, 14.0, 29.0],
    "temperature_2m_min": [18.0, -23.0, 40.0, 1.0, 0.0],
    "precipitation_sum":  [0.75, 0.10,  4.0, 1.25, 2.5],
    "wind_speed_10m_max": [15.0, 10.0, 45.0, 20.0, 33.0],
})

#port values from the metadata_clf "features" column 
new_days = new_days[metadata_clf["features"]]

#validate column names with metadata
assert list(new_days.columns) == metadata_clf["features"], \
"Feature columns do not match metadata!"

predictions = weather_clf.predict(new_days)
probabilities = weather_clf.predict_proba(new_days)[:,1]

#use weather model to test my data
for i, (_, row) in enumerate(new_days.iterrows()):
    #Grabs the prediction and probability for the current day
    prediction = predictions[i]
    probability = probabilities[i]

    label = "good for running" if prediction else "skip"

    print(f"\nDay {i+1}")
    print("-" * 25)
    print(f"temperature_2m_max : {row['temperature_2m_max']} °C")
    print(f"temperature_2m_min : {row['temperature_2m_min']} °C")
    print(f"precipitation_sum  : {row['precipitation_sum']} mm")
    print(f"wind_speed_10m_max : {row['wind_speed_10m_max']} km/h")
    print(f"Prediction         : {label}")
    print(f"Probability(good)  : {probability:.2%}")
    
#Task 03
'''
1. Day 4 is my borderline example because the weather is close to my
  running thresholds. The model predicted approximately 0.52,
  which is only slightly above the default cutoff. I would describe
  this prediction as uncertain. In a real application I would
  probably show a message such as "Conditions are borderline—check
  the detailed forecast before running."

2. If predict_weather.py is run before train_weather_classifier.py,
  the saved model and metadata files will not exist and joblib.load()
  will raise a FileNotFoundError. I would catch this exception and
  display a helpful message telling the user to run
  train_weather_classifier.py first to create the required files.

3. The prediction script would be modified to automatically fetch the
  next day's weather forecast, extract the required features, create a
  DataFrame with those values, and use the trained model to make a prediction.
  '''