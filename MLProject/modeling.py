# trigger workflow

import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='furniture_preprocessed.csv')
args = parser.parse_args()

mlflow.autolog()

df = pd.read_csv(args.data_path)

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    print("Model trained and logged successfully.")
    print(f"MSE: {mse:.2f}, R2 Score: {r2:.2f}")
