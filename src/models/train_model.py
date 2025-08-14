import joblib
import pandas as pd

def main():
    model = joblib.load("model.pkl")
    print("✅ Model loaded.")

    # Example: single prediction
    sample_data = pd.DataFrame([{
        "feature1": 1.2,
        "feature2": 3.4
    }])
    prediction = model.predict(sample_data)
    print("Prediction:", prediction)

if __name__ == "__main__":
    main()
