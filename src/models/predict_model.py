import joblib
from src.eda.eda import run_eda
from src.feature_engineering.feature_eng1 import feature_engineering_1
from src.feature_engineering.feature_eng2 import feature_engineering_2
from src.model_selection.model_selection import select_and_train_model

def main():
    # Load and process data
    df = run_eda()
    df = feature_engineering_1(df)
    df = feature_engineering_2(df)

    # Train model
    model = select_and_train_model(df)

    # Save model
    joblib.dump(model, "model.pkl")
    print("✅ Model saved as model.pkl")

if __name__ == "__main__":
    main()
