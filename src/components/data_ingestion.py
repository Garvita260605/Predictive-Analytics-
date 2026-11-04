
import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException


class DataIngestion:

    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "raw.csv")
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            df = pd.read_csv("data/raw/Airlines Dataset.csv")

            logging.info(f"Dataset loaded: {df.shape}")

            df = df[
                (df["cancelled"] == 0) &
                (df["diverted"] == 0)
            ].copy()

            df = df.dropna(subset=["arr_delay"])

            df["delay_status"] = (
                df["arr_delay"] > 15
            ).astype(int)

            feature_cols = [
                "year",
                "month",
                "day_of_month",
                "day_of_week",
                "op_unique_carrier",
                "origin",
                "dest",
                "crs_dep_time",
                "crs_arr_time",
                "crs_elapsed_time",
                "distance"
            ]

            df = df[feature_cols + ["delay_status"]]

            os.makedirs("artifacts", exist_ok=True)

            df.to_csv(
                self.raw_data_path,
                index=False
            )

            logging.info("Splitting dataset")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["delay_status"]
            )

            train_set.to_csv(
                self.train_data_path,
                index=False
            )

            test_set.to_csv(
                self.test_data_path,
                index=False
            )

            logging.info("Data ingestion completed")

            print("Data ingestion completed successfully!")
            print("Raw dataset shape:", df.shape)
            print("Training dataset shape:", train_set.shape)
            print("Testing dataset shape:", test_set.shape)

            return self.train_data_path, self.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
