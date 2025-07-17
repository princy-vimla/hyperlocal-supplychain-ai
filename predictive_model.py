import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from data_collection import integrate_data

# Load or generate training data
def load_training_data(cities, locations):
    dfs = []
    for city, loc in zip(cities, locations):
        df = integrate_data(city, loc)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Train model
def train_model(df):
    X = df[['sentiment', 'weather', 'historical_sales']]
    y = df['historical_sales'] * (1 + df['sentiment'] * 0.1 + df['weather'] * 0.01)  # Simulated target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Predict demand
def predict_demand(model, new_data):
    return model.predict(new_data[['sentiment', 'weather', 'historical_sales']])

# Main execution (for testing)
if __name__ == "__main__":
    cities = ["Mumbai", "Delhi"]
    locations = ["19.0760,72.8777,10mi", "28.7041,77.1068,10mi"]
    df = load_training_data(cities, locations)
    model, X_test, y_test = train_model(df)
    predictions = predict_demand(model, X_test)
    print(f"Predictions: {predictions}")