import os
import sys
import json
import warnings
import joblib
import pandas as pd
warnings.filterwarnings("ignore")

def predict_sample(sample_dict=None):
    model_path = os.path.join("models", "rf_model.joblib")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run `python train.py` first.")
        sys.exit(1)
        
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    feature_stats = bundle["feature_stats"]
    
    if sample_dict is None:
        print("No input provided. Using Healthy Control cohort median as baseline test sample:")
        sample_dict = feature_stats["hc_median"]
    
    input_df = pd.DataFrame([sample_dict])[feature_cols]
    pred = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    
    print("\n" + "="*50)
    print("PARKINSON'S SPEECH PREDICTION RESULT")
    print("="*50)
    if pred == 1:
        print(f"Outcome:        POSITIVE FOR PARKINSON'S (Class 1)")
    else:
        print(f"Outcome:        HEALTHY / NEGATIVE (Class 0)")
        
    print(f"PD Probability: {probs[1]*100:.2f}%")
    print(f"HC Probability: {probs[0]*100:.2f}%")
    print("="*50)
    
    return pred, probs

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1], "r") as f:
            data = json.load(f)
        predict_sample(data)
    else:
        predict_sample()
