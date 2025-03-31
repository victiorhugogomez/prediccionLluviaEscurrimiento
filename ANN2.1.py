import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import os

def load_and_clean_data(file_paths):
    dfs = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df = df[["Date", "Streamflow", "p_ens"]]
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.dropna(inplace=True)
    return combined_df

def normalize_data(df):
    scaler = StandardScaler()
    df[["Streamflow", "p_ens"]] = scaler.fit_transform(df[["Streamflow", "p_ens"]])
    return df, scaler

def calculate_lag(df):
    precip = df["p_ens"].values
    streamflow = df["Streamflow"].values
    lags = np.arange(-len(precip) + 1, len(precip))
    correlation = np.correlate(precip - np.mean(precip), streamflow - np.mean(streamflow), mode="full")
    optimal_lag = lags[np.argmax(correlation)]
    return optimal_lag

def adjust_data_by_lag(df, lag):
    df["p_ens_lagged"] = df["p_ens"].shift(lag)
    df["p_ens_lagged_2"] = df["p_ens"].shift(lag + 1)
    df["p_ens_lagged_3"] = df["p_ens"].shift(lag + 2)
    df.dropna(inplace=True)
    return df

def train_model(df):
    X = df[["p_ens_lagged", "p_ens_lagged_2", "p_ens_lagged_3"]].values
    y = df["Streamflow"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, MAE: {mae}")

    return model, scaler

def save_predictions(df, model, scaler):
    X = df[["p_ens_lagged", "p_ens_lagged_2", "p_ens_lagged_3"]].values
    y_real = df["Streamflow"].values

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled).flatten()

    results_df = pd.DataFrame({
        "Date": df["Date"],
        "Streamflow_real": y_real,
        "Streamflow_predicho": predictions
    })

    results_df.to_csv("resultados_prediccion_mejorado.csv", index=False)
    print("Archivo guardado como resultados_prediccion_mejorado.csv. Ábrelo en Excel para verlo.")

def plot_results(df, model, scaler):
    X = df[["p_ens_lagged", "p_ens_lagged_2", "p_ens_lagged_3"]].values
    y = df["Streamflow"].values

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], y, label="Valores Reales", color="green")
    plt.plot(df["Date"], predictions, label="Predicciones", color="red")
    plt.xlabel("Fecha")
    plt.ylabel("Escurrimiento")
    plt.title("Comparación de Escurrimiento Real vs Predicho")
    plt.legend()
    plt.grid()
    plt.show()

    save_predictions(df, model, scaler)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_paths = [
        os.path.join(script_dir, "merged_data_cabra1.xlsx"),
        os.path.join(script_dir, "merged_data_cabra2.xlsx"),
        os.path.join(script_dir, "merged_data_cabra3.xlsx"),
        os.path.join(script_dir, "merged_data_cabra5.xlsx"),
    ]
    df = load_and_clean_data(file_paths)
    df, scaler = normalize_data(df)
    lag = calculate_lag(df)
    df = adjust_data_by_lag(df, lag)
    model, scaler = train_model(df)
    plot_results(df, model, scaler)
