import numpy as np
import pandas as pd

def clean_data_train(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the Ryanair TOW training dataset:
    - Replaces '(null)' with np.nan
    - Converts DepartureDate to datetime
    - Converts relevant columns to numeric
    - Drops rows with missing ActualTOW or FLownPassengers
    """
    df = df.copy()
    df.replace("(null)", np.nan, inplace=True)

    if "DepartureDate" in df.columns:
        df["DepartureDate"] = pd.to_datetime(df["DepartureDate"], dayfirst=True, errors="coerce")

    numeric_cols = [
        "ActualTOW", "FLownPassengers", "BagsCount", "FlightBagsWeight",
        "ActualFlightTime", "ActualTotalFuel"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    must_have = ["ActualTOW", "FLownPassengers"]
    df.dropna(subset=[col for col in must_have if col in df.columns], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def clean_data_val(df: pd.DataFrame, median_flown_passengers=None) -> pd.DataFrame:
    """
    Cleans the Ryanair TOW validation dataset:
    - Replaces '(null)' with np.nan
    - Converts DepartureDate to datetime
    - Converts relevant columns to numeric
    - Imputes missing FLownPassengers with the provided median (from train)
    - Drops rows with missing ActualTOW (if present)
    """
    df = df.copy()
    df.replace("(null)", np.nan, inplace=True)

    if "DepartureDate" in df.columns:
        df["DepartureDate"] = pd.to_datetime(df["DepartureDate"], dayfirst=True, errors="coerce")

    numeric_cols = [
        "ActualTOW", "FLownPassengers", "BagsCount", "FlightBagsWeight",
        "ActualFlightTime", "ActualTotalFuel"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing FLownPassengers with median from train
    if "FLownPassengers" in df.columns and median_flown_passengers is not None:
        df["FLownPassengers"] = df["FLownPassengers"].fillna(median_flown_passengers)

    # Drop rows with missing ActualTOW (if present in validation)
    if "ActualTOW" in df.columns:
        df.dropna(subset=["ActualTOW"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df

def handle_outliers(df: pd.DataFrame, cap_dict: dict) -> pd.DataFrame:
    """
    Clips outliers in specified columns at the provided lower and upper limits.
    cap_dict: dict of {col: (low, high)} where low is the 1st percentile and high is the 99th percentile from TRAIN.
    """
    df = df.copy()
    for col, (low, high) in cap_dict.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=low, upper=high)
    return df

def impute_data(df: pd.DataFrame, bags_count_col="BagsCount", bags_weight_col="FlightBagsWeight",
                bags_median=None, avg_bag_weight=None) -> pd.DataFrame:
    """
    Imputes missing values:
    - BagsCount: impute with median
    - FlightBagsWeight: impute as BagsCount * avg bag weight (using rows where both are present)
    """
    df = df.copy()

    if bags_count_col in df.columns and bags_median is not None:
        df[bags_count_col] = df[bags_count_col].fillna(bags_median)

    if bags_weight_col in df.columns and bags_count_col in df.columns and avg_bag_weight is not None:
        missing_mask = df[bags_weight_col].isna() & df[bags_count_col].notna()
        df.loc[missing_mask, bags_weight_col] = df.loc[missing_mask, bags_count_col] * avg_bag_weight

    return df

def preprocess_train_val(df_train_raw, df_val_raw):
    # Clean train
    df_train_clean = clean_data_train(df_train_raw)
    # Compute median for FLownPassengers (for val imputation)
    median_flown_passengers = df_train_clean["FLownPassengers"].median()

    # Clean validation (impute FLownPassengers with train median)
    df_val_clean = clean_data_val(df_val_raw, median_flown_passengers=median_flown_passengers)

    # Outlier capping: fit caps on train, apply to both train and val
    cap_cols = ["BagsCount", "FlightBagsWeight", "ActualFlightTime"]
    cap_dict = {
        col: (
            df_train_clean[col].quantile(0.01),
            df_train_clean[col].quantile(0.99)
        )
        for col in cap_cols if col in df_train_clean.columns
    }
    df_train_clean = handle_outliers(df_train_clean, cap_dict)
    df_val_clean = handle_outliers(df_val_clean, cap_dict)

    # Imputation stats from train
    bags_median = df_train_clean["BagsCount"].median() if "BagsCount" in df_train_clean.columns else None
    if "BagsCount" in df_train_clean.columns and "FlightBagsWeight" in df_train_clean.columns:
        avg_bag_weight = (df_train_clean["FlightBagsWeight"] / df_train_clean["BagsCount"]).mean()
    else:
        avg_bag_weight = None

    # Impute train
    df_train_clean = impute_data(df_train_clean, bags_median=bags_median, avg_bag_weight=avg_bag_weight)

    # Impute val using train statistics
    df_val_clean = impute_data(df_val_clean, bags_median=bags_median, avg_bag_weight=avg_bag_weight)

    return df_train_clean, df_val_clean