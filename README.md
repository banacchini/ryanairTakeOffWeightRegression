# Ryanair Take-Off Weight Regression

## Overview

This project predicts the take-off weight of Ryanair flights using machine learning. It provides a robust data preprocessing pipeline, exploratory data analysis (EDA) in Jupyter notebooks, and a reproducible workflow for training and validation.

## Features

- **Data Cleaning & Preprocessing:**  
  - Outlier capping based on training set percentiles  
  - Imputation of missing values using statistics  
  - Consistent transformations for both training and validation sets  
- **Exploratory Data Analysis:**  
  - Correlation analysis  
  - Visualization of key features and missing data patterns  
- **Prediction Pipeline:**  
  - Preprocessed data is used to train regression models  
  - Predictions are saved to `predictions.csv`  

## File Structure

- `notebooks/` — Jupyter notebooks for EDA and preprocessing, and modeling  
- `preprocessing/preprocessing.py` — Core functions for data cleaning and feature engineering  
- `predictions.csv` — Model output for the validation set  
- `data/` — Raw datasets
- `environment.yml` — Conda environment specification  
- `readme.md` — Project documentation  

## Installation

1. **Clone the repository** and navigate to the project folder.
2. **Create the Conda environment** with all required dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate <your_env_name>
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
