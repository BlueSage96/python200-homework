import joblib
import json 

#Load pipeline
weather_clf = joblib.load("models/weather_classifier.pkl")
print("Weather:", weather_clf)

#Load metadata
with open("models/weather_classifier_metadata.json",'r') as f:
    metadata_clf = json.load(f)

print(f"\nTask 01:\n")
print(f"City: {metadata_clf['city']}")
print(f"Features: {metadata_clf['features']}")
print(f"Test AUC: {metadata_clf['test_auc']}")