import joblib

def load_model(path: str):
    """Load the trained ML model."""
    return joblib.load(path)

def predict_model(model, features):
    """Return model prediction."""
    return model.predict(features)
