import joblib
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from category_encoders import CatBoostEncoder
from xgboost import XGBClassifier
from make_dataset import load_dataset

def build_pipeline(num_cols, cat_cols):
    pipe_num = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    pipe_cat = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', CatBoostEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('numerical', pipe_num, num_cols),
        ('categorical', pipe_cat, cat_cols)
    ])
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', model)
    ])
    return pipeline

def train_and_evaluate():
    df = load_dataset("../data/multisim_dataset.parquet")
    X = df.drop('target', axis=1)
    y = df['target']

    num_cols = X.select_dtypes(include='number').columns.to_list()
    cat_cols = X.select_dtypes(include='object').columns.to_list()

    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X, y)

    joblib.dump(pipeline, "../models/model.pkl")
    print("✅ Model saved at ../models/model.pkl")

    # Cross-validation
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X, y, cv=kf, scoring=scoring)

    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"{metric.capitalize()} Mean: {scores.mean():.4f}")

if __name__ == "__main__":
    train_and_evaluate()
