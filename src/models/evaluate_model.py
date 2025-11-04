import pandas as pd
import pickle
import json

from sklearn.metrics import mean_squared_error, r2_score

config: dict[str, str] = {
    "DATA_PROCESSED": "data/processed",
    "DATA_RAW": "data/raw",
}

# Load the datasets
try:
    X_test = pd.read_csv(f"{config['DATA_PROCESSED']}/X_test_scaled.csv")
    y_test = pd.read_csv(f"{config['DATA_PROCESSED']}/y_test.csv")
except FileNotFoundError:
    raise Exception("failed to load datasets X_test.csv and y_test.csv")

# Drop the date column from X_test
X_test_no_date = X_test.drop(columns=["date"])

# Load the trained model
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test_no_date)

# Create a dataframe with predictions
predictions_df = pd.DataFrame(predictions, columns=["predictions"])

# Add the date column to the predictions dataframe
predictions_df.insert(0, "date", X_test["date"])

# Save the predictions
predictions_df.to_csv("data/processed/predictions.csv", index=False)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Create a dictionary with the scores
scores = {"mse": mse, "r2": r2}

# Save the scores to a JSON file
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)

print(f"Predictions saved to {config['DATA_PROCESSED']}/predictions.csv")
print(f"Evaluation scores saved to metrics/scores.json: {scores}")
