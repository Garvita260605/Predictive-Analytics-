
import pickle
import pandas as pd

print("Checking saved preprocessing object...")

with open("artifacts/preprocessor.pkl", "rb") as f:
    reloaded = pickle.load(f)

print("Preprocessor loaded successfully!")

test_df = pd.read_csv("artifacts/test.csv")

X_test = test_df.drop(columns=["delay_status"])

transformed_test = reloaded.transform(X_test)

print("\nReload Verification Completed!")

print("Original test shape:", X_test.shape)
print("Transformed test shape:", transformed_test.shape)

print("\nSaved preprocessing pipeline is working!")
