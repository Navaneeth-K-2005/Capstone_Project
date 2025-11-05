import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(df: pd.DataFrame, target: str, features: list):
    X = df[features]
    y = df[target]
    model = LinearRegression()
    model.fit(X, y)
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model

def load_model():
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, X: pd.DataFrame):
    return model.predict(X)
