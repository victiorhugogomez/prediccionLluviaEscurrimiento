
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pyswarms as ps
import matplotlib.pyplot as plt
import random
import itertools
from tkinter import Tk, filedialog

# === Selector de archivo ===
def select_file(title):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Excel files", "*.xlsx")])
    return file_path

# === Cargar y normalizar datos ===
file_path = select_file("Selecciona el archivo de datos climáticos")
print("Cargando y normalizando datos...")

df = pd.read_excel(file_path).dropna()
df_scaled = df.copy()
for col in df.columns:
    df_scaled[col] = df[col] / df[col].max()

# === Preparar datos ===
def prepare_data(df, feature_cols, target_col, look_back=10):
    X, y = [], []
    for i in range(len(df) - look_back):
        X.append(df[feature_cols].iloc[i:i+look_back].values)
        y.append(df[target_col].iloc[i+look_back])
    return np.array(X), np.array(y)

# === Métricas adicionales ===
def nash_sutcliffe_efficiency(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def sum_squared_errors(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)

# === Función para construir y entrenar modelo ===
def build_and_train_model(X, y, units, learning_rate, batch_size, epochs):
    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(units, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X).flatten()
    return y_pred

# === Evaluar métricas ===
def evaluate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    nse = nash_sutcliffe_efficiency(y_true, y_pred)
    sse = sum_squared_errors(y_true, y_pred)
    return mse, mae, nse, sse

# === PSO Optimization ===
def evaluate_model_pso(params):
    results = []
    for p in params:
        units = int(p[0])
        learning_rate = p[1]
        batch_size = max(8, int(p[2]))
        epochs = int(p[3])
        y_pred = build_and_train_model(X_p, y_p, units, learning_rate, batch_size, epochs)
        mse = mean_squared_error(y_p, y_pred)
        results.append(mse)
    return np.array(results)

# === Random Search ===
def random_search(n_iter=10):
    best_score = float('inf')
    best_params = None
    for i in range(n_iter):
        print(f"Random Search iteración {i+1}/{n_iter}...")
        units = random.randint(10, 100)
        lr = 10**random.uniform(-4, -2)
        batch = random.choice([8, 16, 32, 64])
        epochs = random.randint(10, 100)
        y_pred = build_and_train_model(X_p, y_p, units, lr, batch, epochs)
        mse = mean_squared_error(y_p, y_pred)
        if mse < best_score:
            best_score = mse
            best_params = (units, lr, batch, epochs)
    return best_params

# === Grid Search ===
def grid_search():
    units_list = [20, 50, 100]
    lr_list = [0.001, 0.005]
    batch_list = [16, 32]
    epoch_list = [30, 50]

    best_score = float('inf')
    best_params = None
    total = len(units_list) * len(lr_list) * len(batch_list) * len(epoch_list)
    count = 1

    for units, lr, batch, epochs in itertools.product(units_list, lr_list, batch_list, epoch_list):
        print(f"Grid Search combinación {count}/{total}...")
        y_pred = build_and_train_model(X_p, y_p, units, lr, batch, epochs)
        mse = mean_squared_error(y_p, y_pred)
        if mse < best_score:
            best_score = mse
            best_params = (units, lr, batch, epochs)
        count += 1
    return best_params

# === Configurar experimento: lluvia-escurrimiento ===
print("Preparando datos (lluvia-escurrimiento)...")
X_p, y_p = prepare_data(df_scaled, ['p_ens'], 'Streamflow')

# --- PSO
print("Ejecutando PSO...")
bounds = ([10, 0.0001, 8, 10], [100, 0.01, 64, 100])
optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=4, options={'c1': 1.5, 'c2': 1.5, 'w': 0.7}, bounds=bounds)
best_cost_pso, best_pos_pso = optimizer.optimize(evaluate_model_pso, iters=5)
units_pso, lr_pso, batch_pso, ep_pso = int(best_pos_pso[0]), best_pos_pso[1], int(best_pos_pso[2]), int(best_pos_pso[3])
y_pred_pso = build_and_train_model(X_p, y_p, units_pso, lr_pso, batch_pso, ep_pso)
metrics_pso = evaluate_metrics(y_p, y_pred_pso)

# --- Random Search
print("Ejecutando Random Search...")
units_r, lr_r, batch_r, ep_r = random_search()
y_pred_r = build_and_train_model(X_p, y_p, units_r, lr_r, batch_r, ep_r)
metrics_r = evaluate_metrics(y_p, y_pred_r)

# --- Grid Search
print("Ejecutando Grid Search...")
units_g, lr_g, batch_g, ep_g = grid_search()
y_pred_g = build_and_train_model(X_p, y_p, units_g, lr_g, batch_g, ep_g)
metrics_g = evaluate_metrics(y_p, y_pred_g)

# === Mostrar resultados
print("\n--- Resultados comparativos (lluvia-escurrimiento) ---")
print(f"PSO       -> MSE: {metrics_pso[0]:.4f}, MAE: {metrics_pso[1]:.4f}, NSE: {metrics_pso[2]:.4f}, SSE: {metrics_pso[3]:.4f}")
print(f"Random    -> MSE: {metrics_r[0]:.4f}, MAE: {metrics_r[1]:.4f}, NSE: {metrics_r[2]:.4f}, SSE: {metrics_r[3]:.4f}")
print(f"Grid      -> MSE: {metrics_g[0]:.4f}, MAE: {metrics_g[1]:.4f}, NSE: {metrics_g[2]:.4f}, SSE: {metrics_g[3]:.4f}")

# === Graficar
plt.figure(figsize=(12,6))
plt.plot(y_p, label='Medido', color='black')
plt.plot(y_pred_pso, label='PSO', linestyle='--')
plt.plot(y_pred_r, label='Random Search', linestyle='-.')
plt.plot(y_pred_g, label='Grid Search', linestyle=':')
plt.title("Comparación de Predicciones - Lluvia-Escurrimiento")
plt.xlabel("Tiempo")
plt.ylabel("Escurrimiento Normalizado")
plt.legend()
plt.tight_layout()
plt.savefig("grafica_resultados.png")
plt.show(block=True)
