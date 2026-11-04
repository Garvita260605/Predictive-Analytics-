
import os
import sys
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException


class DataTransformation:

    def __init__(self):
        self.preprocessor_path = os.path.join(
            "artifacts", "preprocessor.pkl"
        )

    def get_data_transformer(self, numeric_cols, categorical_cols):
        try:

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_cols),
                    ("cat", categorical_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:

            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_col = "delay_status"

            drop_cols = [target_col]

            numeric_cols = (
                train_df.drop(columns=drop_cols)
                .select_dtypes(include=["number"])
                .columns.tolist()
            )

            categorical_cols = (
                train_df.drop(columns=drop_cols)
                .select_dtypes(include=["object", "category"])
                .columns.tolist()
            )

            logging.info(f"Numeric columns: {numeric_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            print("Numerical columns:", numeric_cols)
            print("Categorical columns:", categorical_cols)

            preprocessor = self.get_data_transformer(
                numeric_cols,
                categorical_cols
            )

            X_train = train_df.drop(columns=drop_cols)
            y_train = train_df[target_col]

            X_test = test_df.drop(columns=drop_cols)
            y_test = test_df[target_col]

            X_train_arr = preprocessor.fit_transform(X_train)

            X_test_arr = preprocessor.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)

            with open(self.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info("Preprocessor saved successfully")

            print("Data transformation completed successfully!")
            print("Transformed training shape:", X_train_arr.shape)
            print("Transformed testing shape:", X_test_arr.shape)

            return X_train_arr, y_train, X_test_arr, y_test

        except Exception as e:
            raise CustomException(e, sys)
