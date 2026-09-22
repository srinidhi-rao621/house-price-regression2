# House Price Linear Regression

Multiple linear regression model that predicts house prices from square footage,
number of bedrooms, and number of bathrooms.

## Quick Start

```bash
pip install -r requirements.txt
python model.py
```

## What it does

1. Loads `house_prices.csv` (50 labelled houses)
2. Splits data 80/20 into train / test sets
3. Fits a `LinearRegression` model (scikit-learn)
4. Prints coefficients and evaluation metrics
5. Predicts the price of a sample house (2000 sqft / 3 bed / 2 bath)
6. Saves `plots.png` with five diagnostic charts

## Model

**Multiple Linear Regression** (Ordinary Least Squares)

```
Price = intercept + beta1*sqft + beta2*bedrooms + beta3*bathrooms
```

Fitted on the full 50-sample dataset:

| Parameter   | Value            |
|-------------|-----------------|
| Intercept   | -$31,127         |
| β sqft      | $169.24 / sqft   |
| β bedrooms  | $6,424           |
| β bathrooms | $9,005           |
| R²          | 0.9898           |
| RMSE        | $12,355          |
| MAE         | $8,084           |

## Files

| File               | Description                        |
|--------------------|------------------------------------|
| `model.py`         | Training, evaluation & plots       |
| `house_prices.csv` | 50-row labelled dataset            |
| `requirements.txt` | Python dependencies                |
| `README.md`        | This file                          |
