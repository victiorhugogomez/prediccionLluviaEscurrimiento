
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Add, GlobalAveragePooling1D
)
import tensorflow as tf
import optuna
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# -------------------------------------
# Funciones base
# -------------------------------------

def select_file(title):
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("Excel Files", "*.xlsx")])

def cargar_datos(filepath, ventana_dias):
    df = pd.read_excel(filepath)
    df = df[['Year', 'Month', 'Day', 'p_ens', 'Streamflow']].dropna()
    df['fecha'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.sort_values('fecha')
    df['dayofyear_norm'] = df['fecha'].dt.dayofyear / 365.0
    df['p_ens_norm'] = df['p_ens'] / df['p_ens'].max()
    df['Streamflow_norm'] = df['Streamflow'] / df['Streamflow'].max()
    X, y = [], []
    for i in range(len(df) - ventana_dias):
        lluvia = df['p_ens_norm'].iloc[i:i+ventana_dias].values
        dia = df['dayofyear_norm'].iloc[i:i+ventana_dias].values
        secuencia = np.stack([lluvia, dia], axis=-1)
        objetivo = df['Streamflow_norm'].iloc[i+ventana_dias]
        X.append(secuencia)
        y.append(objetivo)
    return np.array(X), np.array(y)

class PositionalEncodingLearnable(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = self.add_weight(name="pos_embed", shape=(1, seq_len, d_model))

    def call(self, x):
        return x + self.pos_embedding

def graficar(y_true, y_pred, mse, mae, sse, nse, elapsed, ventana_dias):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Medido', color='blue')
    plt.plot(y_pred, label='Predicho', color='red', linestyle='dashed')
    plt.xlabel('Tiempo')
    plt.ylabel('Escurrimiento')
    plt.title(f'Modelo Transformer Optimizado - {ventana_dias} dias\n'
              f'MSE: {mse:.4f}, MAE: {mae:.4f}, SSE: {sse:.4f}, NSE: {nse:.4f}, Tiempo: {elapsed:.2f}s')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------
# Funcion objetivo para Optuna
# -------------------------------------

def objective(trial):
    
    ventana_dias = trial.suggest_int('ventana_dias', 15, 120) # Numero de dias que se usaran como ventana de entrada para predecir el escurrimiento    
    num_heads = trial.suggest_categorical('num_heads', [2, 4, 8]) # Numero de cabezas de atencion en la capa MultiHeadAttention    
    key_dim = trial.suggest_categorical('key_dim', [8, 16, 32]) # Dimension del vector clave (key) en la atencion; afecta el tama;o de proyeccion interna    
    num_layers = trial.suggest_int('num_layers', 1, 4) # Numero de capas del bloque Transformer (atencion + normalizacion)
    dense_units = trial.suggest_categorical('dense_units', [64, 128, 256])  # Numero de unidades en las capas densas del modelo (antes de la salida)   
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)  # Porcentaje de neuronas que se "apagan" aleatoriamente para evitar sobreajuste    
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32]) # Cantidad de muestras procesadas antes de actualizar los pesos del modelo    
    epochs = 20 # Numero de epocas (veces que el modelo vera todos los datos de entrenamiento)

    X, y = cargar_datos(filepath_global, ventana_dias)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs = Input(shape=X.shape[1:]) # entrada del modelo (los datos que se usan para predecir)
    x = Dense(dense_units)(inputs) # capa que ayuda a procesar los datos
    x = PositionalEncodingLearnable(X.shape[1], dense_units)(x)  # agrega informacion sobre el orden de los datos

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)  # busca las partes importantes de los datos
        x = Add()([x, attn_output]) # combina la salida con la entrada anterior
        x = LayerNormalization()(x) # ajusta los valores para que sean mas estables normalizando 
        x = Dropout(dropout_rate)(x) # apaga partes del modelo para evitar errores

    x = GlobalAveragePooling1D()(x) # resume toda la informacion en un solo valor
    x = Dense(dense_units, activation='relu')(x) # otra capa para seguir procesando los datos
    x = Dropout(0.2)(x) # vuelve a apagar una parte del modelo para hacerlo mas robusto
    outputs = Dense(1)(x) # capa final que da el resultado (la prediccion)

    model = Model(inputs, outputs)  # crea el modelo completo
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) # prepara el modelo para entrenarlo

    model.fit(X_train, y_train, # entrena el modelo con los datos
              validation_data=(X_test, y_test),
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)

    y_pred = model.predict(X_test).flatten() # usa el modelo para hacer predicciones
    mse = mean_squared_error(y_test, y_pred) # mide el error cuadrado promedio
    mae = mean_absolute_error(y_test, y_pred) # mide el error promedio normal
    sse = np.sum((y_test - y_pred) ** 2) # suma total de los errores

    return mse + mae + 0.1 * sse # devuelve una medida del error total

# -------------------------------------
# Ejecucion principal
# -------------------------------------

def main():
    global filepath_global
    filepath_global = select_file("Selecciona el archivo de datos")
    if not filepath_global:
        print("No se selecciono archivo.")
        return

    print("🚀 Iniciando optimizacion de hiperparametros...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("\n🏆 Mejor configuracion:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    # Entrenamiento final con mejores hiperparametros
    best = study.best_params
    ventana_dias = best['ventana_dias']
    X, y = cargar_datos(filepath_global, ventana_dias)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    inputs = Input(shape=X.shape[1:])
    x = Dense(best['dense_units'])(inputs)
    x = PositionalEncodingLearnable(X.shape[1], best['dense_units'])(x)
    for _ in range(best['num_layers']):
        attn_output = MultiHeadAttention(num_heads=best['num_heads'], key_dim=best['key_dim'])(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        x = Dropout(best['dropout_rate'])(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(best['dense_units'], activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    print("\n🧠 Entrenando modelo final con mejores parametros...")
    start = time.time()
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=40,
              batch_size=best['batch_size'],
              verbose=1)

    y_pred = model.predict(X_test).flatten()
    elapsed = time.time() - start

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    sse = np.sum((y_test - y_pred) ** 2)
    nse = 1 - (sse / np.sum((y_test - np.mean(y_test)) ** 2))

    graficar(y_test, y_pred, mse, mae, sse, nse, elapsed, ventana_dias)

if __name__ == "__main__":
    main()
