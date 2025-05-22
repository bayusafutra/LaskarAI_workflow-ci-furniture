import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Tambahkan parser untuk menerima path dataset dari MLProject
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='furniture_preprocessed.csv')
args = parser.parse_args()

# Aktifkan autolog dari MLflow
mlflow.autolog()

# Muat dataset dari argumen
df = pd.read_csv(args.data_path)

# Pisahkan fitur dan target
X = df.drop(columns=["price"])
y = df["price"]

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
model = LinearRegression()

# Mulai experiment MLflow
with mlflow.start_run():
    model.fit(X_train, y_train)

    # Prediksi pada data uji
    y_pred = model.predict(X_test)

    # Evaluasi model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Catat manual juga (meskipun autolog sudah mencatat ini)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    print("Model trained and logged successfully.")
    print(f"MSE: {mse:.2f}, R2 Score: {r2:.2f}")
