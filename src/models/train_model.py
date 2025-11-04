import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor

config: dict[str, str] = {
    "DATA_PROCESSED": "data/processed",
    "DATA_RAW": "data/raw",
}

# Load the datasets
try:
    X_train = pd.read_csv(f"{config['DATA_PROCESSED']}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{config['DATA_PROCESSED']}/y_train.csv")
except FileNotFoundError:
    raise Exception("failed to load datasets X_train.csv and y_train.csv")

# Drop the date column from X_train
X_train = X_train.drop(columns=["date"])

# Load the best parameters
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Initialize the model with the best parameters
rf = RandomForestRegressor(**best_params, random_state=42)

# Train the model
rf.fit(X_train, y_train.values.ravel())

# Save the trained model
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model trained and saved to models/trained_model.pkl")
