import pandas as pd
import joblib
from preprocessing.preprocessing import preprocess_train_val

# Load raw training and validation data
df_train_raw = pd.read_csv("data/training.csv", sep="\t")
df_val_raw = pd.read_csv("data/validation.csv", sep="\t")

# Preprocess (clean, impute, cap outliers)
df_train_clean, df_val_clean = preprocess_train_val(df_train_raw, df_val_raw)

# Load the trained CatBoost model
model = joblib.load("trained_models/catboost_best_model.pkl")

# Predict TOW
y_pred = model.predict(df_val_clean)

# Save predictions to CSV
output = df_val_clean.copy()
output["PredictedTOW"] = y_pred
output[["PredictedTOW"]].to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")