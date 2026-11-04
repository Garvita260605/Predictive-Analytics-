# ================================================================
# FLIGHT DELAY PREDICTION
# UNIT 3 - COMPLETE EXPLORATORY DATA ANALYSIS (EDA)
# ================================================================

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# ================================================================
# 1. CONFIGURATION
# ================================================================

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Airlines Dataset.csv"
)

# Output folders
PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

EDA_OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "eda"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EDA_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 2. HELPER FUNCTION
# ================================================================

def save_plot(filename):
    """
    Save the current matplotlib figure.
    """

    path = EDA_OUTPUT_DIR / filename

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Plot saved: {path}")


# ================================================================
# 3. START
# ================================================================

print("\n")
print("=" * 80)
print("FLIGHT DELAY PREDICTION - UNIT 3")
print("COMPLETE EXPLORATORY DATA ANALYSIS")
print("=" * 80)


# ================================================================
# 4. CHECK DATASET
# ================================================================

print("\n[1] CHECKING DATASET PATH")
print("-" * 80)

print("Project root:")
print(PROJECT_ROOT)

print("\nDataset path:")
print(DATA_PATH)

if not DATA_PATH.exists():

    print("\nERROR: Dataset not found!")

    print("\nExpected location:")
    print(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "Airlines Dataset.csv"
    )

    print("\nFiles currently inside data/raw:")

    raw_folder = (
        PROJECT_ROOT
        / "data"
        / "raw"
    )

    if raw_folder.exists():

        for file in raw_folder.iterdir():
            print(" -", file.name)

    sys.exit(1)


print("\nDataset found successfully!")


# ================================================================
# 5. LOAD DATASET
# ================================================================

print("\n[2] LOADING DATASET")
print("-" * 80)

df = pd.read_csv(
    DATA_PATH,
    low_memory=False
)

print("Dataset loaded successfully.")

print(
    f"Rows    : {df.shape[0]:,}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ================================================================
# 6. FIRST FIVE ROWS
# ================================================================

print("\n[3] FIRST FIVE ROWS")
print("-" * 80)

print(
    df.head().to_string()
)


# ================================================================
# 7. LAST FIVE ROWS
# ================================================================

print("\n[4] LAST FIVE ROWS")
print("-" * 80)

print(
    df.tail().to_string()
)


# ================================================================
# 8. DATASET SHAPE
# ================================================================

print("\n[5] DATASET SHAPE")
print("-" * 80)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ================================================================
# 9. COLUMN NAMES
# ================================================================

print("\n[6] COLUMN NAMES")
print("-" * 80)

for number, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{number:02d}. {column}"
    )


# ================================================================
# 10. DATA TYPES
# ================================================================

print("\n[7] DATA TYPES")
print("-" * 80)

dtype_df = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str).values
})

print(
    dtype_df.to_string(index=False)
)


# ================================================================
# 11. DATASET INFORMATION
# ================================================================

print("\n[8] DATASET INFORMATION")
print("-" * 80)

df.info()


# ================================================================
# 12. NUMERICAL STATISTICS
# ================================================================

print("\n[9] NUMERICAL STATISTICS")
print("-" * 80)

numeric_summary = df.describe().T

print(
    numeric_summary.to_string()
)

numeric_summary.to_csv(
    EDA_OUTPUT_DIR
    / "numerical_summary.csv"
)


# ================================================================
# 13. CATEGORICAL STATISTICS
# ================================================================

print("\n[10] CATEGORICAL STATISTICS")
print("-" * 80)

categorical_summary = (
    df.describe(
        include="object"
    ).T
)

print(
    categorical_summary.to_string()
)

categorical_summary.to_csv(
    EDA_OUTPUT_DIR
    / "categorical_summary.csv"
)


# ================================================================
# 14. MISSING VALUES
# ================================================================

print("\n[11] MISSING VALUE ANALYSIS")
print("-" * 80)

missing_df = pd.DataFrame({

    "Missing_Count":
        df.isnull().sum(),

    "Missing_Percentage":
        df.isnull().mean() * 100
})

missing_df = (
    missing_df
    .sort_values(
        "Missing_Percentage",
        ascending=False
    )
)

print(
    missing_df.to_string()
)

missing_df.to_csv(
    EDA_OUTPUT_DIR
    / "missing_values.csv"
)


# ================================================================
# 15. MISSING VALUE PLOT
# ================================================================

missing_nonzero = (
    missing_df[
        missing_df["Missing_Count"] > 0
    ]
)

if not missing_nonzero.empty:

    plt.figure(
        figsize=(12, 8)
    )

    sns.barplot(
        x=missing_nonzero[
            "Missing_Percentage"
        ],
        y=missing_nonzero.index
    )

    plt.title(
        "Missing Values by Column"
    )

    plt.xlabel(
        "Missing Percentage"
    )

    plt.ylabel(
        "Column"
    )

    save_plot(
        "01_missing_values.png"
    )


# ================================================================
# 16. DUPLICATES
# ================================================================

print("\n[12] DUPLICATE ANALYSIS")
print("-" * 80)

duplicate_count = (
    df.duplicated().sum()
)

duplicate_percentage = (
    duplicate_count
    / len(df)
    * 100
)

print(
    "Duplicate rows:",
    duplicate_count
)

print(
    "Duplicate percentage:",
    round(
        duplicate_percentage,
        2
    ),
    "%"
)


# ================================================================
# 17. CANCELLATION ANALYSIS
# ================================================================

print("\n[13] CANCELLATION ANALYSIS")
print("-" * 80)

cancelled_counts = (
    df["cancelled"]
    .value_counts()
)

print(
    cancelled_counts.to_string()
)

print("\nPercentage:")

print(
    (
        df["cancelled"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2).to_string()
)


plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=df,
    x="cancelled"
)

plt.title(
    "Flight Cancellation Distribution"
)

plt.xlabel(
    "Cancelled"
)

plt.ylabel(
    "Number of Flights"
)

plt.xticks(
    [0, 1],
    [
        "Not Cancelled",
        "Cancelled"
    ]
)

save_plot(
    "02_cancellation_distribution.png"
)


# ================================================================
# 18. DIVERSION ANALYSIS
# ================================================================

print("\n[14] DIVERSION ANALYSIS")
print("-" * 80)

diverted_counts = (
    df["diverted"]
    .value_counts()
)

print(
    diverted_counts.to_string()
)

print("\nPercentage:")

print(
    (
        df["diverted"]
        .value_counts(
            normalize=True
        )
        * 100
    ).round(2).to_string()
)


# ================================================================
# 19. ARRIVAL DELAY STATISTICS
# ================================================================

print("\n[15] ARRIVAL DELAY ANALYSIS")
print("-" * 80)

print(
    df["arr_delay"]
    .describe()
    .to_string()
)


# ================================================================
# 20. ARRIVAL DELAY DISTRIBUTION
# ================================================================

plt.figure(
    figsize=(12, 6)
)

sns.histplot(
    df["arr_delay"].dropna(),
    bins=100
)

plt.title(
    "Distribution of Arrival Delay"
)

plt.xlabel(
    "Arrival Delay (minutes)"
)

plt.ylabel(
    "Number of Flights"
)

# Remove extreme visual distortion
plt.xlim(
    df["arr_delay"].quantile(0.01),
    df["arr_delay"].quantile(0.99)
)

save_plot(
    "03_arrival_delay_distribution.png"
)


# ================================================================
# 21. CREATE MODELING DATASET
# ================================================================

print("\n[16] CREATING CLEAN MODELING DATASET")
print("-" * 80)

df_model = df.copy()


# Remove cancelled flights
df_model = df_model[
    df_model["cancelled"] == 0
]


# Remove diverted flights
df_model = df_model[
    df_model["diverted"] == 0
]


# Arrival delay must be available
df_model = df_model.dropna(
    subset=["arr_delay"]
)


print(
    "Original records:",
    f"{len(df):,}"
)

print(
    "Usable records:",
    f"{len(df_model):,}"
)


# ================================================================
# 22. CREATE TARGET
# ================================================================

print("\n[17] CREATING TARGET VARIABLE")
print("-" * 80)

# 1 = delayed more than 15 minutes
# 0 = not delayed more than 15 minutes

df_model["is_delayed"] = (
    df_model["arr_delay"] > 15
).astype(int)


print(
    "Target definition:"
)

print(
    "0 = Arrival delay <= 15 minutes"
)

print(
    "1 = Arrival delay > 15 minutes"
)


# ================================================================
# 23. TARGET DISTRIBUTION
# ================================================================

target_counts = (
    df_model["is_delayed"]
    .value_counts()
)

target_percentages = (
    df_model["is_delayed"]
    .value_counts(
        normalize=True
    )
    * 100
)


print("\nTarget counts:")

print(
    target_counts.to_string()
)

print("\nTarget percentages:")

print(
    target_percentages
    .round(2)
    .to_string()
)


# ================================================================
# 24. TARGET PLOT
# ================================================================

plt.figure(
    figsize=(7, 5)
)

sns.countplot(
    data=df_model,
    x="is_delayed"
)

plt.title(
    "Flight Delay Target Distribution"
)

plt.xlabel(
    "Flight Status"
)

plt.ylabel(
    "Number of Flights"
)

plt.xticks(
    [0, 1],
    [
        "Not Delayed",
        "Delayed > 15 min"
    ]
)

save_plot(
    "04_target_distribution.png"
)


# ================================================================
# 25. FEATURE ENGINEERING
# ================================================================

print("\n[18] FEATURE ENGINEERING")
print("-" * 80)


# Scheduled departure hour
df_model["dep_hour"] = (
    df_model["crs_dep_time"] // 100
)


# Scheduled arrival hour
df_model["arr_hour"] = (
    df_model["crs_arr_time"] // 100
)


print(
    "Created features:"
)

print(
    " - dep_hour"
)

print(
    " - arr_hour"
)


# ================================================================
# 26. DELAY BY AIRLINE
# ================================================================

print("\n[19] DELAY RATE BY AIRLINE")
print("-" * 80)

carrier_delay = (
    df_model
    .groupby(
        "op_unique_carrier"
    )["is_delayed"]
    .agg(
        [
            "mean",
            "count"
        ]
    )
)

carrier_delay.columns = [
    "delay_rate",
    "flight_count"
]

carrier_delay[
    "delay_percentage"
] = (
    carrier_delay[
        "delay_rate"
    ] * 100
)

carrier_delay = (
    carrier_delay
    .sort_values(
        "delay_rate",
        ascending=False
    )
)

print(
    carrier_delay.to_string()
)

carrier_delay.to_csv(
    EDA_OUTPUT_DIR
    / "carrier_delay_rate.csv"
)


plt.figure(
    figsize=(12, 6)
)

sns.barplot(
    data=carrier_delay.reset_index(),
    x="op_unique_carrier",
    y="delay_percentage"
)

plt.title(
    "Flight Delay Rate by Airline"
)

plt.xlabel(
    "Operating Carrier"
)

plt.ylabel(
    "Delay Rate (%)"
)

plt.xticks(
    rotation=45
)

save_plot(
    "05_delay_by_airline.png"
)


# ================================================================
# 27. DELAY BY MONTH
# ================================================================

print("\n[20] DELAY RATE BY MONTH")
print("-" * 80)

monthly_delay = (
    df_model
    .groupby("month")["is_delayed"]
    .mean()
    * 100
)

print(
    monthly_delay.to_string()
)


plt.figure(
    figsize=(8, 5)
)

monthly_delay.plot(
    kind="bar"
)

plt.title(
    "Flight Delay Rate by Month"
)

plt.xlabel(
    "Month"
)

plt.ylabel(
    "Delay Rate (%)"
)

save_plot(
    "06_delay_by_month.png"
)


# ================================================================
# 28. DELAY BY DAY OF WEEK
# ================================================================

print("\n[21] DELAY RATE BY DAY OF WEEK")
print("-" * 80)

weekday_delay = (
    df_model
    .groupby(
        "day_of_week"
    )["is_delayed"]
    .mean()
    * 100
)

print(
    weekday_delay.to_string()
)


plt.figure(
    figsize=(9, 5)
)

weekday_delay.plot(
    kind="bar"
)

plt.title(
    "Flight Delay Rate by Day of Week"
)

plt.xlabel(
    "Day of Week"
)

plt.ylabel(
    "Delay Rate (%)"
)

save_plot(
    "07_delay_by_day_of_week.png"
)


# ================================================================
# 29. DELAY BY DEPARTURE HOUR
# ================================================================

print("\n[22] DELAY RATE BY DEPARTURE HOUR")
print("-" * 80)

hourly_delay = (
    df_model
    .groupby(
        "dep_hour"
    )["is_delayed"]
    .mean()
    * 100
)

print(
    hourly_delay.to_string()
)


plt.figure(
    figsize=(12, 6)
)

hourly_delay.plot(
    kind="line",
    marker="o"
)

plt.title(
    "Flight Delay Rate by Scheduled Departure Hour"
)

plt.xlabel(
    "Departure Hour"
)

plt.ylabel(
    "Delay Rate (%)"
)

plt.xticks(
    range(0, 24)
)

plt.grid(True)

save_plot(
    "08_delay_by_departure_hour.png"
)


# ================================================================
# 30. DELAY BY ORIGIN AIRPORT
# ================================================================

print("\n[23] DELAY RATE BY ORIGIN AIRPORT")
print("-" * 80)

origin_delay = (
    df_model
    .groupby(
        "origin"
    )["is_delayed"]
    .agg(
        [
            "mean",
            "count"
        ]
    )
)

origin_delay.columns = [
    "delay_rate",
    "flight_count"
]

origin_delay[
    "delay_percentage"
] = (
    origin_delay[
        "delay_rate"
    ] * 100
)


# Ignore airports with too few flights
origin_filtered = (
    origin_delay[
        origin_delay[
            "flight_count"
        ] >= 500
    ]
    .sort_values(
        "delay_rate",
        ascending=False
    )
)

print(
    origin_filtered.head(15).to_string()
)


plt.figure(
    figsize=(12, 7)
)

top_origin = (
    origin_filtered
    .head(15)
    .sort_values(
        "delay_rate"
    )
)

sns.barplot(
    x=top_origin[
        "delay_percentage"
    ],
    y=top_origin.index
)

plt.title(
    "Top Origin Airports by Delay Rate"
)

plt.xlabel(
    "Delay Rate (%)"
)

plt.ylabel(
    "Origin Airport"
)

save_plot(
    "09_delay_by_origin.png"
)


# ================================================================
# 31. DELAY BY DESTINATION AIRPORT
# ================================================================

print("\n[24] DELAY RATE BY DESTINATION AIRPORT")
print("-" * 80)

destination_delay = (
    df_model
    .groupby(
        "dest"
    )["is_delayed"]
    .agg(
        [
            "mean",
            "count"
        ]
    )
)

destination_delay.columns = [
    "delay_rate",
    "flight_count"
]

destination_delay[
    "delay_percentage"
] = (
    destination_delay[
        "delay_rate"
    ] * 100
)

destination_filtered = (
    destination_delay[
        destination_delay[
            "flight_count"
        ] >= 500
    ]
    .sort_values(
        "delay_rate",
        ascending=False
    )
)

print(
    destination_filtered
    .head(15)
    .to_string()
)


plt.figure(
    figsize=(12, 7)
)

top_destination = (
    destination_filtered
    .head(15)
    .sort_values(
        "delay_rate"
    )
)

sns.barplot(
    x=top_destination[
        "delay_percentage"
    ],
    y=top_destination.index
)

plt.title(
    "Top Destination Airports by Delay Rate"
)

plt.xlabel(
    "Delay Rate (%)"
)

plt.ylabel(
    "Destination Airport"
)

save_plot(
    "10_delay_by_destination.png"
)


# ================================================================
# 32. DISTANCE VS ARRIVAL DELAY
# ================================================================

print("\n[25] DISTANCE VS ARRIVAL DELAY")
print("-" * 80)

sample_size = min(
    50000,
    len(df_model)
)

sample_df = (
    df_model.sample(
        n=sample_size,
        random_state=42
    )
)


plt.figure(
    figsize=(10, 6)
)

sns.scatterplot(
    data=sample_df,
    x="distance",
    y="arr_delay",
    alpha=0.25
)

plt.title(
    "Distance vs Arrival Delay"
)

plt.xlabel(
    "Distance"
)

plt.ylabel(
    "Arrival Delay (minutes)"
)

plt.ylim(
    df_model[
        "arr_delay"
    ].quantile(0.01),
    df_model[
        "arr_delay"
    ].quantile(0.99)
)

save_plot(
    "11_distance_vs_arrival_delay.png"
)


# ================================================================
# 33. DEPARTURE DELAY VS ARRIVAL DELAY
# ================================================================

print("\n[26] DEPARTURE DELAY VS ARRIVAL DELAY")
print("-" * 80)

plt.figure(
    figsize=(10, 6)
)

sns.scatterplot(
    data=sample_df,
    x="dep_delay",
    y="arr_delay",
    alpha=0.25
)

plt.title(
    "Departure Delay vs Arrival Delay"
)

plt.xlabel(
    "Departure Delay (minutes)"
)

plt.ylabel(
    "Arrival Delay (minutes)"
)

plt.ylim(
    df_model[
        "arr_delay"
    ].quantile(0.01),
    df_model[
        "arr_delay"
    ].quantile(0.99)
)

save_plot(
    "12_departure_vs_arrival_delay.png"
)


# ================================================================
# 34. CORRELATION ANALYSIS
# ================================================================

print("\n[27] CORRELATION ANALYSIS")
print("-" * 80)

numeric_columns = [

    "year",

    "month",

    "day_of_month",

    "day_of_week",

    "op_carrier_fl_num",

    "crs_dep_time",

    "dep_delay",

    "taxi_out",

    "crs_arr_time",

    "arr_delay",

    "crs_elapsed_time",

    "actual_elapsed_time",

    "air_time",

    "distance",

    "carrier_delay",

    "weather_delay",

    "nas_delay",

    "security_delay",

    "late_aircraft_delay"
]


available_numeric_columns = [

    column

    for column in numeric_columns

    if column in df_model.columns
]


correlation = (
    df_model[
        available_numeric_columns
    ]
    .corr()
)


correlation.to_csv(
    EDA_OUTPUT_DIR
    / "correlation_matrix.csv"
)


plt.figure(
    figsize=(16, 13)
)

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Matrix"
)

save_plot(
    "13_correlation_matrix.png"
)


# ================================================================
# 35. CORRELATION WITH ARRIVAL DELAY
# ================================================================

print("\n[28] CORRELATION WITH ARRIVAL DELAY")
print("-" * 80)

arr_delay_correlation = (
    correlation[
        "arr_delay"
    ]
    .sort_values(
        ascending=False
    )
)

print(
    arr_delay_correlation.to_string()
)


# ================================================================
# 36. OUTLIER ANALYSIS
# ================================================================

print("\n[29] OUTLIER ANALYSIS")
print("-" * 80)

plt.figure(
    figsize=(12, 5)
)

sns.boxplot(
    x=df_model[
        "arr_delay"
    ]
)

plt.title(
    "Arrival Delay Outlier Analysis"
)

plt.xlabel(
    "Arrival Delay (minutes)"
)

plt.xlim(
    df_model[
        "arr_delay"
    ].quantile(0.01),

    df_model[
        "arr_delay"
    ].quantile(0.99)
)

save_plot(
    "14_arrival_delay_boxplot.png"
)


# ================================================================
# 37. EXTREME DELAYS
# ================================================================

print("\n[30] EXTREME DELAY RECORDS")
print("-" * 80)

extreme_columns = [

    "fl_date",

    "op_unique_carrier",

    "origin",

    "dest",

    "dep_delay",

    "arr_delay"
]


available_extreme_columns = [

    column

    for column in extreme_columns

    if column in df_model.columns
]


extreme_delays = (
    df_model[
        available_extreme_columns
    ]
    .sort_values(
        "arr_delay",
        ascending=False
    )
    .head(20)
)


print(
    extreme_delays.to_string(
        index=False
    )
)


extreme_delays.to_csv(
    EDA_OUTPUT_DIR
    / "extreme_delays.csv",
    index=False
)


# ================================================================
# 38. PRE-DEPARTURE FEATURES
# ================================================================

print("\n[31] PRE-DEPARTURE FEATURES")
print("-" * 80)

pre_departure_features = [

    "year",

    "month",

    "day_of_month",

    "day_of_week",

    "op_unique_carrier",

    "op_carrier_fl_num",

    "origin",

    "origin_city_name",

    "origin_state_nm",

    "dest",

    "dest_city_name",

    "dest_state_nm",

    "crs_dep_time",

    "crs_arr_time",

    "crs_elapsed_time",

    "distance",

    "dep_hour"
]


print(
    "Features available before departure:"
)

for feature in pre_departure_features:

    print(
        "  +",
        feature
    )


# ================================================================
# 39. DATA LEAKAGE FEATURES
# ================================================================

print("\n[32] DATA LEAKAGE ANALYSIS")
print("-" * 80)

leakage_features = [

    "dep_time",

    "dep_delay",

    "taxi_out",

    "wheels_off",

    "wheels_on",

    "taxi_in",

    "arr_time",

    "actual_elapsed_time",

    "air_time",

    "carrier_delay",

    "weather_delay",

    "nas_delay",

    "security_delay",

    "late_aircraft_delay"
]


print(
    "These variables will NOT be used as "
    "pre-flight prediction inputs:"
)

for feature in leakage_features:

    if feature in df_model.columns:

        print(
            "  -",
            feature
        )


# ================================================================
# 40. FINAL MODELING DATASET
# ================================================================

print("\n[33] CREATING FINAL MODELING DATASET")
print("-" * 80)


final_columns = (
    pre_departure_features
    + ["is_delayed"]
)


final_model_df = (
    df_model[
        final_columns
    ]
    .copy()
)


print(
    "Final modeling dataset shape:"
)

print(
    final_model_df.shape
)


print(
    "\nFinal columns:"
)

for column in final_model_df.columns:

    print(
        "  +",
        column
    )


# ================================================================
# 41. FINAL MISSING VALUES
# ================================================================

print("\n[34] FINAL DATASET MISSING VALUES")
print("-" * 80)

final_missing = pd.DataFrame({

    "Missing_Count":
        final_model_df.isnull().sum(),

    "Missing_Percentage":
        final_model_df.isnull().mean() * 100

})


print(
    final_missing.to_string()
)


# ================================================================
# 42. SAVE FINAL PROCESSED DATA
# ================================================================

print("\n[35] SAVING PROCESSED DATASET")
print("-" * 80)

OUTPUT_PATH = (
    PROCESSED_DIR
    / "flight_delay_eda.csv"
)


final_model_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "Processed dataset saved successfully:"
)

print(
    OUTPUT_PATH
)


# ================================================================
# 43. SAVE TARGET SUMMARY
# ================================================================

target_summary = pd.DataFrame({

    "Class": [
        0,
        1
    ],

    "Meaning": [
        "Not delayed <= 15 minutes",
        "Delayed > 15 minutes"
    ],

    "Count": [
        (
            df_model[
                "is_delayed"
            ] == 0
        ).sum(),

        (
            df_model[
                "is_delayed"
            ] == 1
        ).sum()
    ],

    "Percentage": [
        (
            df_model[
                "is_delayed"
            ] == 0
        ).mean() * 100,

        (
            df_model[
                "is_delayed"
            ] == 1
        ).mean() * 100
    ]

})


target_summary.to_csv(
    EDA_OUTPUT_DIR
    / "target_summary.csv",
    index=False
)


# ================================================================
# 44. SAVE DATASET SUMMARY
# ================================================================

dataset_summary = pd.DataFrame({

    "Metric": [

        "Original Rows",

        "Original Columns",

        "Usable Flights",

        "Cancelled Flights",

        "Diverted Flights",

        "Duplicate Rows",

        "Delayed Flights",

        "Not Delayed Flights",

        "Model Features"

    ],

    "Value": [

        len(df),

        len(df.columns),

        len(df_model),

        (
            df["cancelled"] == 1
        ).sum(),

        (
            df["diverted"] == 1
        ).sum(),

        duplicate_count,

        (
            df_model[
                "is_delayed"
            ] == 1
        ).sum(),

        (
            df_model[
                "is_delayed"
            ] == 0
        ).sum(),

        len(pre_departure_features)

    ]

})


dataset_summary.to_csv(
    EDA_OUTPUT_DIR
    / "dataset_summary.csv",
    index=False
)


# ================================================================
# 45. FINAL SUMMARY
# ================================================================

delayed_count = (
    df_model[
        "is_delayed"
    ] == 1
).sum()


not_delayed_count = (
    df_model[
        "is_delayed"
    ] == 0
).sum()


delayed_percentage = (
    delayed_count
    / len(df_model)
    * 100
)


not_delayed_percentage = (
    not_delayed_count
    / len(df_model)
    * 100
)


print("\n")
print("=" * 80)
print("UNIT 3 EDA COMPLETED SUCCESSFULLY")
print("=" * 80)

print(
    f"""
DATASET
-------
Original rows       : {len(df):,}
Original columns    : {len(df.columns)}

CLEANED DATA
------------
Usable flights      : {len(df_model):,}
Cancelled flights   : {(df["cancelled"] == 1).sum():,}
Diverted flights    : {(df["diverted"] == 1).sum():,}
Duplicate rows      : {duplicate_count:,}

TARGET
------
Not delayed         : {not_delayed_count:,}
Not delayed (%)     : {not_delayed_percentage:.2f}%

Delayed >15 min     : {delayed_count:,}
Delayed (%)         : {delayed_percentage:.2f}%

FEATURES
--------
Pre-departure features : {len(pre_departure_features)}

OUTPUT
------
Processed dataset:
{OUTPUT_PATH}

EDA plots:
{EDA_OUTPUT_DIR}

Unit 3 is complete.
"""
)

print("=" * 80)
print("NEXT STEP: UNIT 4 - DATA INGESTION & TRANSFORMATION")
print("=" * 80)
