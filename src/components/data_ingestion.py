import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging


def load_raw_data(data_path: str) -> pd.DataFrame:
    try:
        logging.info("Starting data ingestion for Flight Delay Prediction")

        df = pd.read_csv(data_path)

        logging.info(
            f"Successfully loaded flight dataset with "
            f"{df.shape[0]} rows and {df.shape[1]} columns"
        )

        return df

    except Exception as e:
        logging.error("Error occurred while loading flight dataset")
        raise CustomException(e, sys)


if __name__ == "__main__":
    data_path = os.path.join(
        "data",
        "raw",
        "Airlines Dataset.csv"
    )

    df = load_raw_data(data_path)

    print("Data ingestion successful!")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
