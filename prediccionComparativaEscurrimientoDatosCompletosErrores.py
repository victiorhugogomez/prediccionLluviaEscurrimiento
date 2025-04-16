import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tkinter import Tk, filedialog
import time  # 🕒

def select_file(title):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Excel Files", "*.xlsx")])
    return file_path

def prepare_data(df, feature_cols, target_col, look_back=10):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[feature_cols].iloc[i:i+look_back].values)
        y.append(df[target_col].iloc[i+look_back])
    return np.array(X), np.array(y)

def train_lstm(X_train, y_train):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
    return model

def calculate_errors(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    sse = np.sum((y_true - y_pred.flatten())**2)
    nse = 1 - (np.sum((y_true - y_pred.flatten())**2) / np.sum((y_true - np.mean(y_true))**2))
    return mse, mae, sse, nse

def plot_results(y_true, y_pred, title, mse, mae, sse, nse):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='Medido', color='blue')
    plt.plot(y_pred, label='Predicho', color='red', linestyle='dashed')
    plt.xlabel('Tiempo')
    plt.ylabel('Escurrimiento')
    plt.legend()
    plt.title(f"{title}\nMSE: {mse:.4f}, MAE: {mae:.4f}, SSE: {sse:.4f}, NSE: {nse:.4f}")
    plt.show()

def main():
    print("Selecciona el archivo de datos combinados")
    data_file = select_file("Selecciona el archivo de datos combinados")
    
    if not data_file:
        print("No se seleccionó ningún archivo. Terminando.")
        return
    
    df = pd.read_excel(data_file)
    df = df.dropna()
    
    # Normalizar los datos dividiendo por su valor máximo
    df_scaled = df.copy()
    for col in df.columns:
        df_scaled[col] = df[col] / df[col].max()
    
    # 🕒 Tiempo para modelo con solo precipitación
    start_time_prec = time.time()
    X_prec, y_prec = prepare_data(df_scaled, ['p_ens'], 'Streamflow')
    model_prec = train_lstm(X_prec, y_prec)
    y_pred_prec = model_prec.predict(X_prec)
    elapsed_prec = time.time() - start_time_prec
    print(f"🕒 Tiempo de ejecución (precipitación): {elapsed_prec:.2f} segundos")
    mse_prec, mae_prec, sse_prec, nse_prec = calculate_errors(y_prec, y_pred_prec)
    plot_results(
        y_prec, y_pred_prec,
        f"Predicción con Precipitación (Tiempo: {elapsed_prec:.2f} s)",
        mse_prec, mae_prec, sse_prec, nse_prec
    )

    # 🕒 Tiempo para modelo con todos los factores
    start_time_all = time.time()
    feature_cols = ['p_ens', 'tmin_ens', 'tmax_ens', 'rh_ens', 'wnd_ens', 'srad_ens', 'et_ens', 'pet_pm', 'pet_pt', 'pet_hg']
    X_all, y_all = prepare_data(df_scaled, feature_cols, 'Streamflow')
    model_all = train_lstm(X_all, y_all)
    y_pred_all = model_all.predict(X_all)
    elapsed_all = time.time() - start_time_all
    print(f"🕒 Tiempo de ejecución (todos los factores): {elapsed_all:.2f} segundos")
    mse_all, mae_all, sse_all, nse_all = calculate_errors(y_all, y_pred_all)
    plot_results(
        y_all, y_pred_all,
        f"Predicción con Todos los Factores Meteorológicos (Tiempo: {elapsed_all:.2f} s)",
        mse_all, mae_all, sse_all, nse_all
    )

if __name__ == "__main__":
    main()
