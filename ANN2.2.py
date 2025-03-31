import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.stattools import grangercausalitytests

# Cargar y limpiar los datos
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

# Normalización de los datos
def normalize_data(df):
    scaler = StandardScaler()
    df[["Streamflow", "p_ens"]] = scaler.fit_transform(df[["Streamflow", "p_ens"]])
    return df, scaler

# Cálculo del desfase óptimo usando correlación cruzada
def calculate_lag(df):
    precip = df["p_ens"].values
    streamflow = df["Streamflow"].values
    lags = np.arange(-len(precip) + 1, len(precip))
    correlation = np.correlate(precip - np.mean(precip), streamflow - np.mean(streamflow), mode="full")
    optimal_lag = lags[np.argmax(correlation)]
    
    plt.figure(figsize=(10, 5))
    plt.plot(lags, correlation)
    plt.xlabel("Lag")
    plt.ylabel("Correlación")
    plt.title("Correlación en función del desfase")
    plt.grid()
    plt.show()
    
    return optimal_lag

# Ajuste de los datos según el `lag` determinado
def adjust_data_by_lag(df, lag):
    df["p_ens_lagged"] = df["p_ens"].shift(lag)
    df["p_ens_lagged_2"] = df["p_ens"].shift(lag + 1)
    df["p_ens_lagged_3"] = df["p_ens"].shift(lag + 2)
    df.dropna(inplace=True)
    return df

# Modelo de Red Neuronal con LSTM
def train_lstm_model(df):
    X = df[["p_ens_lagged", "p_ens_lagged_2", "p_ens_lagged_3"]].values
    y = df["Streamflow"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=500, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)
    
    y_pred = model.predict(X_test).flatten()
    
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
    
    return model, scaler

# Comparación con Regresión Lineal
def compare_with_linear_regression(df):
    from sklearn.linear_model import LinearRegression
    
    X = df[["p_ens_lagged", "p_ens_lagged_2", "p_ens_lagged_3"]].values
    y = df["Streamflow"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Regresión Lineal - MAE:", mean_absolute_error(y_test, y_pred))
    print("Regresión Lineal - MSE:", mean_squared_error(y_test, y_pred))
    print("Regresión Lineal - R²:", r2_score(y_test, y_pred))

def main():
    file_paths = [
        "merged_data_cabra1.xlsx",
        "merged_data_cabra2.xlsx",
        "merged_data_cabra3.xlsx",
        "merged_data_cabra5.xlsx",
    ]
    df = load_and_clean_data(file_paths)
    df, scaler = normalize_data(df)
    lag = calculate_lag(df)
    df = adjust_data_by_lag(df, lag)
    
    print("Entrenando LSTM...")
    model, scaler = train_lstm_model(df)
    
    print("Comparando con regresión lineal...")
    compare_with_linear_regression(df)
    
if __name__ == "__main__":
    main()
