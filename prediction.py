import pickle
import numpy as np
import pandas as pd

# Load model and encoders
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

def predict_autism(input_dict):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply encoders to categorical columns
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Drop columns not used in training
    input_df.drop(columns=["ID", "age_desc"], inplace=True, errors="ignore")

    # Predict
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    result = "Autism Detected" if prediction[0] == 1 else "No Autism Detected"
    print(f"Prediction : {result}")
    print(f"Confidence : {round(probability * 100, 2)}%")
    return prediction[0]


# ---------- Test Input ----------
sample_input = {
    "A1_Score"        : 1,
    "A2_Score"        : 1,
    "A3_Score"        : 1,
    "A4_Score"        : 1,
    "A5_Score"        : 1,
    "A6_Score"        : 1,
    "A7_Score"        : 1,
    "A8_Score"        : 1,
    "A9_Score"        : 1,
    "A10_Score"       : 1,
    "age"             : 25.0,
    "gender"          : "m",
    "ethnicity"       : "White-European",
    "jaundice"        : "no",
    "austim"          : "yes",
    "contry_of_res"   : "United States",
    "used_app_before" : "no",
    "result"          : 13.5,
    "relation"        : "Self"
}

predict_autism(sample_input)