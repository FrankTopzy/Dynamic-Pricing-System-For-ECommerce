import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from feature_engineering import add_engineered_features


DATA_PATH = "dynamic_pricing_dataset_realistic.csv"
MODEL_PATH = "pricing_model.pkl"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    required = {
        "product_id",
        "current_price",
        "competitor_price",
        "demand_last_week",
        "stock_level",
        "marketing_spend",
        "season",
        "day_of_week",
        "customer_rating",
        "units_sold",
        "optimal_price",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    X = df.drop(columns=["optimal_price"])
    y = df["optimal_price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    numeric_features = [
        "product_id",
        "current_price",
        "competitor_price",
        "demand_last_week",
        "stock_level",
        "marketing_spend",
        "customer_rating",
        "units_sold",
        "price_gap",
        "stock_to_demand_ratio",
        "is_weekend",
    ]
    categorical_features = ["season", "day_of_week"]

    preprocessor = Pipeline(
        steps=[
            ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
            (
                "encode",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", numeric_features),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            categorical_features,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
        ]
    )

    model = HistGradientBoostingRegressor(random_state=42)

    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Saved model pipeline to {MODEL_PATH}")
    print(f"Test MAE: {mae:,.2f}")
    print(f"Test MSE: {mse:,.2f}")
    print(f"Test R2 : {r2:,.4f}")
    print(
        "Target min/max:",
        f"{float(y.min()):,.2f}",
        f"{float(y.max()):,.2f}",
    )


if __name__ == "__main__":
    main()

