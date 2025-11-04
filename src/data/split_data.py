import pandas as pd
import os

from sklearn.model_selection import train_test_split

config: dict[str, str] = {
    "DATA_PROCESSED": "data/processed",
    "DATA_RAW": "data/raw",
}


def main():
    # Create processed data directory if it doesn't exist
    os.makedirs(config["DATA_PROCESSED"], exist_ok=True)

    # Load the dataset
    try:
        df = pd.read_csv(config["DATA_RAW"] + "/raw.csv")
    except FileNotFoundError:
        raise Exception(
            "raw.csv not found. Please ensure the raw data is in 'data/raw_data/raw.csv'"
        )

    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save the datasets
    X_train.to_csv(f"{config['DATA_PROCESSED']}/X_train.csv", index=False)
    X_test.to_csv(f"{config['DATA_PROCESSED']}/X_test.csv", index=False)

    y_train.to_csv(f"{config['DATA_PROCESSED']}/y_train.csv", index=False)
    y_test.to_csv(f"{config['DATA_PROCESSED']}/y_test.csv", index=False)

    print(f"Data split and saved to {config['DATA_PROCESSED']}")


if __name__ == "__main__":
    main()
