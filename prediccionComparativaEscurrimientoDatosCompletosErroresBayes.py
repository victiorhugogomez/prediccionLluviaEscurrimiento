import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tkinter import Tk, filedialog
import time  # 🕒

# -------------------- Funciones utilitarias --------------------

def select_file(title):
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("Excel Files", "*.xlsx")])

def prepare_data(df, feature_cols, target_col, look_back=10):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[feature_cols].iloc[i:i+look_back].values)
        y.append(df[target_col].iloc[i+look_back])
    return np.array(X), np.array(y)

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
    plt.tight_layout()
    plt.show()

# -------------------- Optimización bayesiana --------------------

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

space = [
    Integer(10, 100, name='units'),
    Integer(8, 64, name='batch_size'),
    Integer(10, 100, name='epochs')
]

@use_named_args(space)
def objective(**params):
    units = int(params['units'])
    batch_size = int(params['batch_size'])
    epochs = int(params['epochs'])
    global X_opt, y_opt
    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, input_shape=(X_opt.shape[1], X_opt.shape[2])),
        LSTM(units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_opt, y_opt, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_opt)
    return mean_squared_error(y_opt, y_pred)

def optimize(X, y):
    global X_opt, y_opt
    X_opt, y_opt = X, y
    res = gp_minimize(objective, space, n_calls=15, random_state=0)
    return res

# -------------------- Bloque principal --------------------

def main():
    print("Selecciona el archivo de datos combinados")
    data_file = select_file("Selecciona el archivo de datos combinados")
    
    if not data_file:
        print("No se seleccionó ningún archivo.")
        return

    df = pd.read_excel(data_file)
    df = df.dropna()

    # Normalizar
    df_scaled = df.copy()
    for col in df.columns:
        df_scaled[col] = df[col] / df[col].max()

    # Preparar datos con solo precipitación
    feature_cols = ['p_ens']
    target_col = 'Streamflow'
    X, y = prepare_data(df_scaled, feature_cols, target_col)

    # 🕒 Inicio del temporizador
    start_time = time.time()

    # Optimización bayesiana
    result = optimize(X, y)
    best_units, best_batch, best_epochs = result.x

    print("\nMejores hiperparámetros encontrados:")
    print(f"Unidades LSTM: {best_units}")
    print(f"Batch size: {best_batch}")
    print(f"Epochs: {best_epochs}")
    print(f"Error mínimo (MSE): {result.fun:.4f}")

    # Entrenar modelo con los mejores parámetros
    model = Sequential([
        LSTM(int(best_units), activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(int(best_units), activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=int(best_epochs), batch_size=int(best_batch), verbose=1)

    # 🕒 Fin del temporizador
    elapsed_time = time.time() - start_time
    print(f"\n🕒 Tiempo total de ejecución: {elapsed_time:.2f} segundos")

    # Predicción y evaluación
    y_pred = model.predict(X)
    mse, mae, sse, nse = calculate_errors(y, y_pred)
    plot_results(
        y, y_pred,
        f"Predicción con Precipitación (Bayesiana, Tiempo: {elapsed_time:.2f} s)",
        mse, mae, sse, nse
    )

if __name__ == "__main__":
    main()
