# How the System Was Built

This document describes the end-to-end build process for the Dynamic Pricing Prediction System.

## 1) Project Goal

Build a machine learning system that estimates an optimal selling price from product, competition, demand, inventory, and temporal context.

## 2) Data Preparation

Source file:

- `dynamic_pricing_dataset.csv` (original)
- `dynamic_pricing_dataset_realistic.csv` (newer dataset with larger price ranges)

Dataset includes numeric and categorical variables and the supervised target `optimal_price`.

Key preparation steps in the notebook:

- Load and inspect data
- Check structure and missing values
- Prepare base predictors and target

## 3) Feature Engineering

To improve predictive signal, the following engineered fields are created:

- `price_gap = current_price - competitor_price`
- `stock_to_demand_ratio = stock_level / (demand_last_week + 1)`
- `is_weekend` from `day_of_week`

Categorical variables are transformed using:

- `pd.get_dummies(..., drop_first=True)` for `season` and `day_of_week`

This creates the final model feature space used during training.

## 4) Train/Test Split

The notebook splits data into train and test sets using `train_test_split` with:

- `test_size=0.33`
- `random_state=42`

This enables reproducible and fair out-of-sample evaluation.

## 5) Model Benchmarking

Multiple regressors are trained and compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor

Evaluation metrics:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R2)

The selected exported model in this repository is a `LinearRegression` model.

## 6) Model Serialization

After training, the chosen model is serialized to:

- `pricing_model.pkl`

This artifact is loaded at runtime by the Streamlit app.

## 7) Repeatable Training Script

In addition to the notebook, the repository includes a repeatable training script:

- `train_model.py`

The script:

- loads `dynamic_pricing_dataset_realistic.csv` (by default)
- performs feature engineering
- applies categorical encoding (`OneHotEncoder(handle_unknown="ignore")`)
- trains a regressor
- serializes a full preprocessing + model `Pipeline` to `pricing_model.pkl`

Run from the project root:

```bash
python train_model.py
```

## 8) Streamlit App Integration

The app (`app.py`) implements production-style inference logic:

1. Collect user input through UI controls.
2. Build a one-row dataframe containing the raw input columns.
3. Call `model.predict(...)` on that dataframe.
4. The loaded pipeline performs feature engineering + encoding internally.
5. Optional debug view displays input shape, columns, and values.

## 9) How Predicted Price Is Computed

The deployed `pricing_model.pkl` is loaded in `app.py` and used like:

- build `input_df` (raw inputs)
- `predicted_price = model.predict(input_df)[0]`

Because the saved object is a full sklearn `Pipeline`, it includes the feature transformations from training (feature engineering + one-hot encoding) before the final model predicts.

Important implication:

- The predicted price range is primarily determined by the **training target (`optimal_price`) distribution** in the dataset used to train `pricing_model.pkl`.
- The app can optionally clip predictions to the dataset target min/max as a safety guardrail.

## 10) Issue Encountered and Fix Applied

### Problem

Inference originally passed only 9 features, while the trained model expected 20.

### Fix

- Added missing UI/context fields (`product_id`, `season`, `day_of_week`)
- Added missing engineered field (`is_weekend`)
- Implemented strict feature schema alignment to training columns
- Forced prediction input into a numeric `(1, 20)` array

This removed the feature mismatch error and made inference consistent with training.

## 11) Suggested Next Improvements

- Save preprocessing + model together in one sklearn `Pipeline`
- Version model with metadata (feature list, sklearn version, timestamp)
- Add unit tests for feature builder
- Add model retraining script for repeatable updates
