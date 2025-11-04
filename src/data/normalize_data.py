import pandas as pd
from sklearn.preprocessing import StandardScaler

config: dict[str, str] = {
    "DATA_PROCESSED": "data/processed",
    "DATA_RAW": "data/raw",
}

# Load the datasets
try:
    X_train = pd.read_csv(f"{config['DATA_PROCESSED']}/X_train.csv")
    X_test = pd.read_csv(f"{config['DATA_PROCESSED']}/X_test.csv")
except FileNotFoundError:
    raise Exception("failed to load datasets X_train.csv and X_test.csv")

# Separate the date column
X_train_date = X_train["date"]
X_test_date = X_test["date"]

# Drop the date column for scaling
X_train_numeric = X_train.drop(columns=["date"])
X_test_numeric = X_test.drop(columns=["date"])

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both datasets
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Convert the scaled arrays back to dataframes
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_numeric.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_numeric.columns)

# Add the date column back
X_train_scaled_df.insert(0, "date", X_train_date)
X_test_scaled_df.insert(0, "date", X_test_date)

# Save the scaled datasets
X_train_scaled_df.to_csv(f"{config['DATA_PROCESSED']}/X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv(f"{config['DATA_PROCESSED']}/X_test_scaled.csv", index=False)

print("Data normalized and saved to {config['DATA_PROCESSED']}/")
