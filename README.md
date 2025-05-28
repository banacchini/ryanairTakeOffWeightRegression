# Ryanair Take-Off Weight Regression

## Overview
This project predicts the take-off weight of Ryanair flights using machine learning. It includes a data preprocessing pipeline, exploratory data analysis (EDA) in Jupyter notebooks, and a reproducible workflow for training and validation.

You can find a detailed analysis and modeling report in [report/report.md](https://github.com/banacchini/ryanairTakeOffWeightRegression/blob/main/report/report.md)

The final model is a CatBoost Regressor, selected for its strong performance and ability to handle categorical features natively. The workflow covers outlier capping, missing value imputation, and careful feature engineering. 

## Key Results
- Best Model: CatBoost Regressor (with hyperparameter tuning)
- Final Cross-Validation RMSE: 742.39
- Most Important Features:
  1. Flown Passengers
  2. ActualTotalFuel
  3. ActualFlightTime

## Features

- **Data Cleaning & Preprocessing:**  
  - Outlier capping at 1st/99th percentiles
  - Imputation of missing values (median or derived values)
  - Consistent transformations for both training and validation sets
- **Feature Engineering::**
  - Categorical features: DepartureAirport, ArrivalAirport, Route, FlightNumber, DepartureWeekday
  - Numeric features: ActualTotalFuel, ActualFlightTime, FlownPassengers, BagsCount, FlightBagsWeight
  - Categorical features encoded with LabelEncoder for tree-based models; only numeric features used for linear models
- **Exploratory Data Analysis:**  
  - Correlation analysis  
  - Visualization of key features and missing data patterns  
- **Prediction Pipeline:**  
  - Preprocessed data is used to train regression models  
  - Predictions are saved to `predictions.csv`

## File Structure

- [notebooks/](https://github.com/banacchini/ryanairTakeOffWeightRegression/tree/master/notebooks) — Jupyter notebooks for EDA and preprocessing, and modeling  
- [preprocessing/preprocessing.py](https://github.com/banacchini/ryanairTakeOffWeightRegression/blob/master/preprocessing/preprocessing.py) — Core functions for data cleaning and feature engineering  
- [predictions.csv](https://github.com/banacchini/ryanairTakeOffWeightRegression/blob/master/predictions.csv) — Model output for the validation set  
- [data/](https://github.com/banacchini/ryanairTakeOffWeightRegression/tree/master/data) — Raw datasets
- [environment.yml](https://github.com/banacchini/ryanairTakeOffWeightRegression/blob/master/environment.yml) — Conda environment specification  
- `readme.md` — Project documentation
- [report/report.md](https://github.com/banacchini/ryanairTakeOffWeightRegression/blob/master/report/report.md) - Markdown report on the project.
- [trained_models/](https://github.com/banacchini/ryanairTakeOffWeightRegression/tree/master/trained_models) - Stores serialized trained and tuned machine learning model `catboost_best_model.pkl`

## Reproducibility
  - All experiments use fixed random seeds for reproducibility.
  - Software versions are specified in `environment.yml` and `requirements.txt`.
  - To reproduce results, follow the installation and usage instructions.

## Installation

1. **Clone the repository** and navigate to the project folder.
2. **Create the Conda environment** with all required dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate myenv
   ```
   Replace `<your_env_name>` with the name specified in `environment.yml`.
3. **(Optional)**: For pip users, install dependencies with:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run EDA and Preprocessing:**  
   Open the notebooks in `notebooks/` with Jupyter or PyCharm's notebook support to explore and preprocess the data.

2. **Preprocess Data in Python:**  
   Example usage of `preprocessing.py`:
   ```python
   import pandas as pd
   from preprocessing import preprocess_train_val

   df_train = pd.read_csv('data/training.csv', sep='\t')
   df_val = pd.read_csv('data/validation.csv', sep='\t')
   df_train_clean, df_val_clean = preprocess_train_val(df_train, df_val)
   ```

3. **Generate Predictions:**  
   Use your regression model on the cleaned data and save results to `predictions.csv`.


## Example

See the notebooks in `notebooks/` for step-by-step EDA, and modeling.

## Potential Improvements
  - Add external features (e.g., weather, distance between airports)
  - Explore advanced encoding for categorical variables
  - Further hyperparameter tuning and model ensembling
  - Try additional models (e.g., LightGBM, deep learning)
