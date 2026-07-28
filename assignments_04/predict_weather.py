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
brand_new_day = pd.DataFrame({
    "temperature_2m_max": [32.0, 15.0, 7.0, 14.0, 29.0],
    "temperature_2m_min": [18.0, -23.0, 40.0, 1.0, 0.0],
    "precipitation_sum":  [0.75, 0.10,  4.0, 1.25, 2.5],
    "wind_speed_10m_max": [15.0, 10.0, 45.0, 20.0, 33.0],
})


#make up days, weather conditions (clearly good, clearly bad, & a borderline case (almost good or bad))
predictions = weather_clf.predict(brand_new_day)
probabilities = weather_clf.predict_proba(brand_new_day)[:,1]

#use weather model to test my data
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    label = "good for running" if pred == 1 else "skip"
    print(f"Day {i+1}:  {label} ({prob:.2f}) probability\n")
    
print(f"Conditions {brand_new_day}")

#Task 03
#1. I did a borderline case for all of the features. 
#The probabilities were: Day 1: 10%, day 2: 25%, day 3 99%, day 4: 70%, day 5: 1%
#The model is confident on days 3 and 4 and uncertain on days 1, 2, and 5.
#I would classify 0.52 as "uncertain".

#2. There was no error message when I ran predict_weather.py first.
#It displays old information until I ran train_weather_classifier.py

#3. The prediction script would be modified to automatically fetch the 
# next day's weather, extract any required features, make a DataFrame 
# from those values, and then automatically test the data via the trained model.