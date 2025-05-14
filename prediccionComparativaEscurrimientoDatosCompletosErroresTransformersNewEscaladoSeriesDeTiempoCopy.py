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
    MultiHeadAttention, Add, GlobalAveragePooling1D, Concatenate
)
import tensorflow as tf

# 🔧 Parámetro de configuración principal
VENTANA_DIAS = 90  # reducimos para tener más muestras

def select_file(title):
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("Excel Files", "*.xlsx")])

def cargar_datos(filepath):
    df = pd.read_excel(filepath)
    df = df[['Year', 'Month', 'Day', 'p_ens', 'Streamflow']].dropna()
    df['fecha'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.sort_values('fecha')

    # Variables adicionales
    df['dayofyear_norm'] = df['fecha'].dt.dayofyear / 365.0

    # Normalización
    df['p_ens_norm'] = df['p_ens'] / df['p_ens'].max()
    df['Streamflow_norm'] = df['Streamflow'] / df['Streamflow'].max()

    X, y = [], []
    for i in range(len(df) - VENTANA_DIAS):
        lluvia = df['p_ens_norm'].iloc[i:i+VENTANA_DIAS].values
        dia = df['dayofyear_norm'].iloc[i:i+VENTANA_DIAS].values

        secuencia = np.stack([lluvia, dia], axis=-1)  # forma: (ventana, 2)
        objetivo = df['Streamflow_norm'].iloc[i+VENTANA_DIAS]
        X.append(secuencia)
        y.append(objetivo)

    return np.array(X), np.array(y)

# 🧠 Positional Encoding aprendible
class PositionalEncodingLearnable(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = self.add_weight(name="pos_embed", shape=(1, seq_len, d_model))


    def call(self, x):
        return x + self.pos_embedding

def modelo_transformer(input_shape, num_heads=4, key_dim=16, num_layers=2):
    inputs = Input(shape=input_shape)

    # Proyección a dimensión deseada
    x = Dense(64)(inputs)
    x = PositionalEncodingLearnable(input_shape[0], 64)(x)

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        x = Dropout(0.1)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def graficar(y_true, y_pred, mse, mae, sse, nse, elapsed):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Medido', color='blue')
    plt.plot(y_pred, label='Predicho', color='red', linestyle='dashed')
    plt.xlabel('Tiempo')
    plt.ylabel('Escurrimiento')
    plt.title(f'Modelo Transformer Mejorado - {VENTANA_DIAS} días\n'
              f'MSE: {mse:.4f}, MAE: {mae:.4f}, SSE: {sse:.4f}, NSE: {nse:.4f}, Tiempo: {elapsed:.2f}s')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    archivo = select_file("Selecciona el archivo de datos")
    if not archivo:
        print("No se seleccionó archivo.")
        return

    print("📥 Cargando datos...")
    X, y = cargar_datos(archivo)

    print(f"🔢 Tamaño de entrada: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Entrenando modelo...")
    start = time.time()
    model = modelo_transformer(X.shape[1:])
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=40, batch_size=16, verbose=1)

    y_pred = model.predict(X_test).flatten()
    elapsed = time.time() - start

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    sse = np.sum((y_test - y_pred) ** 2)
    nse = 1 - (sse / np.sum((y_test - np.mean(y_test)) ** 2))

    graficar(y_test, y_pred, mse, mae, sse, nse, elapsed)

if __name__ == "__main__":
    main()
