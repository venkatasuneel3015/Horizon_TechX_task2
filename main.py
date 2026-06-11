# Car Price Prediction

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

# Load Dataset
df = pd.read_csv("Car_Price_Prediction.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# Features
X = df.drop(
    "Price",
    axis=1
)

# Target
y = df["Price"]


# Separate Columns
categorical = [
    "Make",
    "Model",
    "Fuel Type",
    "Transmission"
]

numeric = [
    "Year",
    "Engine Size",
    "Mileage"
]


# Encoding
preprocessor = ColumnTransformer(
    [
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        ),
        (
            "num",
            "passthrough",
            numeric
        )
    ]
)


# Model
model = Pipeline(
    [
        (
            "preprocess",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42
            )
        )
    ]
)


# Split
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)


# Train
model.fit(
    X_train,
    y_train
)


# Predict
y_pred = model.predict(
    X_test
)


# Evaluation
print("\nResults")

print(
    "MAE:",
    round(
        mean_absolute_error(
            y_test,
            y_pred
        ),
        2
    )
)

print(
    "R2 Score:",
    round(
        r2_score(
            y_test,
            y_pred
        ),
        2
    )
)


# Graph
plt.scatter(
    y_test,
    y_pred
)

plt.xlabel(
    "Actual Price"
)

plt.ylabel(
    "Predicted Price"
)

plt.title(
    "Car Price Prediction"
)

plt.show()