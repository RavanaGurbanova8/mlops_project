import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,  StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandas import Series, DataFrame
from xgboost import XGBClassifier
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_validate





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
