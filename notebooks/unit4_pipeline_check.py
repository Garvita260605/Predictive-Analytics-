
import os
import sys

# Add the main project directory to Python path
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

print("Starting Unit 4 Pipeline...")

train_path, test_path = DataIngestion().initiate_data_ingestion()

print("\nData ingestion completed!")

X_train, y_train, X_test, y_test = (
    DataTransformation().initiate_data_transformation(
        train_path,
        test_path
    )
)

print("\nData transformation completed!")

print("\nFinal Results")

print("Training features:", X_train.shape)
print("Training target:", y_train.shape)

print("Testing features:", X_test.shape)
print("Testing target:", y_test.shape)

print("\nUnit 4 Pipeline Executed Successfully!")
