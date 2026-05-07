import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "pricing_model.pkl"
DATA_PATH = PROJECT_ROOT / "dynamic_pricing_dataset_realistic.csv"

# Ensure the feature engineering function is available during unpickling.
# Some pickles may reference it as __main__.add_engineered_features depending on
# how the model was originally serialized.
try:
    import feature_engineering as _fe

    add_engineered_features = _fe.add_engineered_features  # noqa: F401
except Exception:
    def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[no-redef]
        df = df.copy()
        df["price_gap"] = df["current_price"] - df["competitor_price"]
        df["stock_to_demand_ratio"] = df["stock_level"] / (df["demand_last_week"] + 1)
        df["is_weekend"] = df["day_of_week"].isin(["saturday", "sunday"]).astype(int)
        return df

@st.cache_data(show_spinner=False)
def get_training_target_range(path: str = "dynamic_pricing_dataset.csv"):
    try:
        df = pd.read_csv(path, usecols=["optimal_price"])
        min_price = float(df["optimal_price"].min())
        max_price = float(df["optimal_price"].max())
        return min_price, max_price
    except Exception:
        return None


def load_model_with_recovery(model_path: Path):
    try:
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)
    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as exc:
        st.warning(
            "Model file could not be loaded in this environment. "
            "Attempting to retrain and regenerate `pricing_model.pkl`..."
        )
        try:
            from train_model import main as train_main

            train_main()
            with open(model_path, "rb") as model_file:
                return pickle.load(model_file)
        except Exception as retrain_exc:
            st.error(
                "Unable to load or rebuild the model automatically. "
                "Please confirm deployment dependencies are installed "
                "(especially scikit-learn, pandas, numpy)."
            )
            st.exception(retrain_exc)
            st.stop()
        st.stop()


# Load trained model
model = load_model_with_recovery(MODEL_PATH)

st.title("Dynamic Pricing Prediction System")

st.write("Adjust the sliders to predict the optimal price")

@st.cache_data(show_spinner=False)
def get_training_ranges(path: str = str(DATA_PATH)):
    try:
        df = pd.read_csv(
            path,
            usecols=[
                "product_id",
                "current_price",
                "competitor_price",
                "demand_last_week",
                "stock_level",
                "marketing_spend",
                "customer_rating",
                "units_sold",
                "optimal_price",
            ],
        )
        ranges = {col: (float(df[col].min()), float(df[col].max())) for col in df.columns}
        return ranges
    except Exception:
        return None


ranges = get_training_ranges()

# Sliders for inputs
product_id_min, product_id_max = (0, 10000)
if ranges and "product_id" in ranges:
    product_id_min, product_id_max = int(ranges["product_id"][0]), int(ranges["product_id"][1])
product_id_default = int((product_id_min + product_id_max) / 2)
product_id = st.slider("Product ID", int(product_id_min), int(product_id_max), product_id_default)

current_price_min, current_price_max = (0.0, 10000.0)
if ranges and "current_price" in ranges:
    current_price_min, current_price_max = ranges["current_price"]
current_price_default = float((current_price_min + current_price_max) / 2)
current_price = st.slider(
    "Current Price",
    float(current_price_min),
    float(current_price_max),
    float(current_price_default),
)

competitor_price_min, competitor_price_max = (0.0, 10000.0)
if ranges and "competitor_price" in ranges:
    competitor_price_min, competitor_price_max = ranges["competitor_price"]
competitor_price_default = float((competitor_price_min + competitor_price_max) / 2)
competitor_price = st.slider(
    "Competitor Price",
    float(competitor_price_min),
    float(competitor_price_max),
    float(competitor_price_default),
)

demand_min, demand_max = (0, 1000)
if ranges and "demand_last_week" in ranges:
    demand_min, demand_max = int(ranges["demand_last_week"][0]), int(ranges["demand_last_week"][1])
demand_default = int((demand_min + demand_max) / 2)
demand_last_week = st.slider("Demand Last Week", int(demand_min), int(demand_max), demand_default)

stock_min, stock_max = (0, 1000)
if ranges and "stock_level" in ranges:
    stock_min, stock_max = int(ranges["stock_level"][0]), int(ranges["stock_level"][1])
stock_default = int((stock_min + stock_max) / 2)
stock_level = st.slider("Stock Level", int(stock_min), int(stock_max), stock_default)

marketing_spend_min, marketing_spend_max = (0.0, 5000.0)
if ranges and "marketing_spend" in ranges:
    marketing_spend_min, marketing_spend_max = ranges["marketing_spend"]
marketing_spend_default = float((marketing_spend_min + marketing_spend_max) / 2)
marketing_spend = st.slider(
    "Marketing Spend",
    float(marketing_spend_min),
    float(marketing_spend_max),
    float(marketing_spend_default),
)

rating_min, rating_max = (1.0, 5.0)
if ranges and "customer_rating" in ranges:
    rating_min, rating_max = ranges["customer_rating"]
customer_rating_default = float((rating_min + rating_max) / 2)
customer_rating = st.slider(
    "Customer Rating",
    float(rating_min),
    float(rating_max),
    float(customer_rating_default),
)

units_min, units_max = (0, 1000)
if ranges and "units_sold" in ranges:
    units_min, units_max = int(ranges["units_sold"][0]), int(ranges["units_sold"][1])
units_default = int((units_min + units_max) / 2)
units_sold = st.slider("Units Sold", int(units_min), int(units_max), units_default)

season = st.selectbox("Season", ["festival", "monsoon", "summer", "winter"])

day_of_week = st.selectbox(
    "Day of Week",
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
)

def build_raw_input() -> pd.DataFrame:
    # The retrained model is a preprocessing+model pipeline that accepts raw columns.
    return pd.DataFrame(
        [
            {
                "product_id": product_id,
                "current_price": current_price,
                "competitor_price": competitor_price,
                "demand_last_week": demand_last_week,
                "stock_level": stock_level,
                "marketing_spend": marketing_spend,
                "season": season,
                "day_of_week": day_of_week,
                "customer_rating": customer_rating,
                "units_sold": units_sold,
            }
        ]
    )


show_debug = st.checkbox("Show model input debug info", value=False)
use_guardrails = st.checkbox("Use price guardrails (recommended)", value=True)

# Predict button
if st.button("Predict Optimal Price"):
    input_df = build_raw_input()
    if show_debug:
        st.write("Raw model input (before preprocessing):")
        st.dataframe(input_df)

    prediction = model.predict(input_df)
    raw_price = float(prediction[0])

    target_range = None
    if ranges and "optimal_price" in ranges:
        target_range = ranges["optimal_price"]
    clipped_price = raw_price
    if use_guardrails and target_range is not None:
        min_price, max_price = target_range
        clipped_price = float(np.clip(raw_price, min_price, max_price))

        if raw_price < min_price or raw_price > max_price:
            st.warning(
                f"Raw model output ({raw_price:.2f}) is outside the training target range "
                f"({min_price:.2f}–{max_price:.2f}). Showing a clipped value for safety."
            )

    st.success(f"Recommended Price: {clipped_price:.2f}")
    if show_debug:
        st.write(f"Raw prediction: {raw_price:.2f}")
        if target_range is not None:
            st.write(f"Training target range: {target_range[0]:.2f}–{target_range[1]:.2f}")