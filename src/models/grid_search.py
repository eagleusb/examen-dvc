import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

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

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
}

# Initialize the grid search
grid_search = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
)

# Fit the grid search
grid_search.fit(X_train, y_train.values.ravel())

# Get the best parameters
best_params = grid_search.best_params_

# Save the best parameters
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

print(f"Best parameters saved to models/best_params.pkl: {best_params}")
