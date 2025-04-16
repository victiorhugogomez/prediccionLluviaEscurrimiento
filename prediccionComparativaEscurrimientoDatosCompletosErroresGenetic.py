import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tkinter import Tk, filedialog
from geneticalgorithm import geneticalgorithm as ga
from tensorflow.keras import Input

# Seleccionar archivo
def select_file(title):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Excel Files", "*.xlsx")])
    return file_path

# Preparar datos para LSTM
def prepare_data(df, feature_cols, target_col, look_back=10):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[feature_cols].iloc[i:i+look_back].values)
        y.append(df[target_col].iloc[i+look_back])
    return np.array(X), np.array(y)

# Función de error adaptada para algoritmo genético
def objective_function_ga(params):
    units = int(params[0])
    batch_size = int(params[1])
    epochs = int(params[2])

    try:
        model = Sequential([
            Input(shape=(X.shape[1], X.shape[2])),
            LSTM(units, activation='relu', return_sequences=True),
            LSTM(units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_global, y_global, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred = model.predict(X_global)
        mse = mean_squared_error(y_global, y_pred)
    except:
        mse = 1e6  # Penalización por error
    return mse

# Optimización con algoritmo genético
def optimize_lstm_with_ga(X, y):
    global X_global, y_global
    X_global = X
    y_global = y

    varbound = np.array([[10, 100], [8, 64], [10, 100]])  # [units, batch_size, epochs]
    algorithm_param = {
        'max_num_iteration': 20,
        'population_size': 10,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.3,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': None
    }

    model_param = {
        'function': objective_function_ga,
        'dimension': 3,
        'variable_type': 'int',
        'variable_boundaries': varbound
    }

    model = ga(
    function=model_param['function'],
    dimension=model_param['dimension'],
    variable_type=model_param['variable_type'],
    variable_boundaries=model_param['variable_boundaries'],
    algorithm_parameters=algorithm_param,
    function_timeout=600.0  # Extiende tiempo para permitir entrenamiento
    )

    model.run()

    best_params = model.output_dict['variable']
    best_units = int(best_params[0])
    best_batch = int(best_params[1])
    best_epochs = int(best_params[2])

    print(f"\n🧠 Mejores parámetros encontrados por GA:")
    print(f" - Unidades LSTM: {best_units}")
    print(f" - Batch size: {best_batch}")
    print(f" - Épocas: {best_epochs}")

    final_model = Sequential([
        tf.keras.Input(shape=(X.shape[1], X.shape[2])),
        LSTM(best_units, activation='relu', return_sequences=True),
        LSTM(best_units, activation='relu'),
        Dense(1)
    ])
    final_model.compile(optimizer='adam', loss='mse')
    final_model.fit(X, y, epochs=best_epochs, batch_size=best_batch, verbose=1)

    return final_model

# Calcular errores
def calculate_errors(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    sse = np.sum((y_true - y_pred.flatten())**2)
    nse = 1 - (np.sum((y_true - y_pred.flatten())**2) / np.sum((y_true - np.mean(y_true))**2))
    return mse, mae, sse, nse

# Graficar resultados
def plot_results(y_true, y_pred, title, mse, mae, sse, nse):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='Medido', color='blue')
    plt.plot(y_pred, label='Predicho', color='red', linestyle='dashed')
    plt.xlabel('Tiempo')
    plt.ylabel('Escurrimiento')
    plt.legend()
    plt.title(f"{title}\nMSE: {mse:.4f}, MAE: {mae:.4f}, SSE: {sse:.4f}, NSE: {nse:.4f}")
    plt.show()

# Función principal
def main():
    print("Selecciona el archivo de datos combinados")
    data_file = select_file("Selecciona el archivo de datos combinados")

    if not data_file:
        print("No se seleccionó ningún archivo. Terminando.")
        return

    df = pd.read_excel(data_file)
    df = df.dropna()

    # Normalizar datos
    df_scaled = df.copy()
    for col in df.columns:
        df_scaled[col] = df[col] / df[col].max()

    # Predicción con solo precipitación
    X_prec, y_prec = prepare_data(df_scaled, ['p_ens'], 'Streamflow')
    model_prec = optimize_lstm_with_ga(X_prec, y_prec)
    y_pred_prec = model_prec.predict(X_prec)
    mse_prec, mae_prec, sse_prec, nse_prec = calculate_errors(y_prec, y_pred_prec)
    plot_results(y_prec, y_pred_prec, "Predicción con Precipitación", mse_prec, mae_prec, sse_prec, nse_prec)

    # # Predicción con todos los factores meteorológicos (si deseas activarlo)
    # feature_cols = ['p_ens', 'tmin_ens', 'tmax_ens', 'rh_ens', 'wnd_ens', 'srad_ens', 'et_ens', 'pet_pm', 'pet_pt', 'pet_hg']
    # X_all, y_all = prepare_data(df_scaled, feature_cols, 'Streamflow')
    # model_all = optimize_lstm_with_ga(X_all, y_all)
    # y_pred_all = model_all.predict(X_all)
    # mse_all, mae_all, sse_all, nse_all = calculate_errors(y_all, y_pred_all)
    # plot_results(y_all, y_pred_all, "Predicción con Todos los Factores Meteorológicos", mse_all, mae_all, sse_all, nse_all)

# Ejecutar
if __name__ == "__main__":
    main()
