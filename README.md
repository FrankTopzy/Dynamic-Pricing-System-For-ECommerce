# Dynamic Pricing Prediction System

This project predicts an optimal product price using a supervised machine learning model and a Streamlit web app interface.

## System Implementation Overview

The system has three major parts:

1. Data source (`dynamic_pricing_dataset_realistic.csv` or `dynamic_pricing_dataset.csv`)
2. Model development notebook (`Dynamic_pricing.ipynb`) / training script (`train_model.py`)
3. Inference app (`app.py`) and serialized model pipeline (`pricing_model.pkl`)

## Core Problem

Given product and market context (current price, competitor price, demand, stock, marketing spend, timing signals, and quality indicators), predict a recommended optimal price.

## Data and Features

### Base input fields

- `product_id`
- `current_price`
- `competitor_price`
- `demand_last_week`
- `stock_level`
- `marketing_spend`
- `season`
- `day_of_week`
- `customer_rating`
- `units_sold`

Target:

- `optimal_price`

### Engineered features

During training, additional features are created:

- `price_gap = current_price - competitor_price`
- `stock_to_demand_ratio = stock_level / (demand_last_week + 1)`
- `is_weekend = 1 if day_of_week in [saturday, sunday] else 0`

Categorical encoding:

- One-hot encoding on `season` and `day_of_week` with `drop_first=True`

Final training matrix contains 20 input features.

## Modeling

The notebook compares several regressors, including:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Metrics used include:

- MSE
- MAE
- R2

The deployed app loads a serialized **sklearn Pipeline** from `pricing_model.pkl` (preprocessing + model).

## Inference Flow (`app.py`)

1. Collect user inputs via Streamlit sliders/select boxes.
2. Build a one-row dataframe containing the raw input columns.
3. Call `model.predict(...)` on the raw dataframe.
4. The loaded pipeline performs feature engineering + encoding internally, then predicts.
5. Display recommended price.

## How Recommended Price Is Calculated

The app loads `pricing_model.pkl`, which is a preprocessing + model pipeline trained to predict `optimal_price`.

At runtime:

- The app sends the raw inputs as a one-row `pandas.DataFrame` into `model.predict(...)`.
- The pipeline:
  - creates engineered features (`price_gap`, `stock_to_demand_ratio`, `is_weekend`)
  - one-hot encodes `season` and `day_of_week`
  - runs the final regressor to output the predicted price.

Important behavior:

- The predicted price range is driven by the **training target (`optimal_price`) distribution** in your chosen dataset.
- Guardrails in the app can optionally clip outputs to the dataset target min/max range.

## Important Implementation Note

The app explicitly aligns inference input columns with the training schema to avoid feature-count mismatch errors such as:

`ValueError: X has 9 features, but LinearRegression is expecting 20 features as input.`

## Run the App

From the project root:

```bash
python -m streamlit run app.py
```

## Retrain the Model

This project includes a repeatable training script. From the project root:

```bash
python train_model.py
```

By default it trains on `dynamic_pricing_dataset_realistic.csv` and overwrites `pricing_model.pkl`.

## Files in This Project

- `app.py`: Streamlit inference UI (sends raw inputs to the model pipeline)
- `Dynamic_pricing.ipynb`: model training and evaluation workflow
- `train_model.py`: repeatable training script (builds and saves a preprocessing + model pipeline)
- `dynamic_pricing_dataset.csv`: original training dataset
- `dynamic_pricing_dataset_realistic.csv`: newer dataset with larger price ranges
- `pricing_model.pkl`: serialized trained pipeline used by the app
