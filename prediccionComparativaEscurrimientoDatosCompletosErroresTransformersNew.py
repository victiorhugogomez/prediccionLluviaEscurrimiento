import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import Tk, filedialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, GlobalAveragePooling1D
import tensorflow as tf

# Parámetros
VENTANA_DIAS = 30

# 1. Selección de archivo
def select_file(title):
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("Excel Files", "*.xlsx")])

# 2. Carga y preprocesamiento
def cargar_y_preparar_datos(filepath):
    df = pd.read_excel(filepath)
    df = df[['Year', 'Month', 'Day', 'p_ens', 'Streamflow']].dropna()
    df['fecha'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.sort_values('fecha')
    df['dayofyear'] = df['fecha'].dt.dayofyear

    # Normalización
    max_lluvia = df['p_ens'].max()
    max_escurrimiento = df['Streamflow'].max()
    df['p_ens_norm'] = df['p_ens'] / max_lluvia
    df['Streamflow_norm'] = df['Streamflow'] / max_escurrimiento

    secuencias, objetivos = [], []

    for i in range(len(df) - VENTANA_DIAS):
        ventana = df.iloc[i:i+VENTANA_DIAS]
        frase = " ".join([
            f"DIA_{int(row['dayofyear']):03d}_LLUVIA_{row['p_ens_norm']:.4f}"
            for _, row in ventana.iterrows()
        ])
        secuencias.append(frase)
        objetivos.append(df.iloc[i+VENTANA_DIAS]['Streamflow_norm'])  # Día siguiente

    return secuencias, np.array(objetivos), max_escurrimiento
# imprimir la secuencia 

# 3. Tokenización
def tokenizar(secuencias, max_len, vocab_size=10000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(secuencias)
    sec_int = tokenizer.texts_to_sequences(secuencias)
    sec_pad = pad_sequences(sec_int, maxlen=max_len, padding='post', truncating='post')
    return tokenizer, sec_pad

# 4. Modelo Transformer encoder simple
def transformer_encoder_model(vocab_size, embedding_dim=64, max_len=VENTANA_DIAS):
    inputs = Input(shape=(max_len,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = LayerNormalization()(x)
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embedding_dim)(x, x)
    x = Dropout(0.1)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 5. Gráfico
def graficar(y_true, y_pred, mse, mae, sse, nse, elapsed):
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Medido', color='blue')
    plt.plot(y_pred, label='Predicho', color='red', linestyle='dashed')
    plt.xlabel('Muestra')
    plt.ylabel('Escurrimiento')
    plt.title(f'Predicción con Transformer (Ventana: {VENTANA_DIAS} días)\nMSE: {mse:.2f}, MAE: {mae:.2f}, SSE: {sse:.2f}, NSE: {nse:.2f} | Tiempo: {elapsed:.2f}s')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 6. Main
def main():
    archivo = select_file("Selecciona el archivo de datos meteorológicos")
    if not archivo:
        print("No se seleccionó archivo.")
        return

    print("📥 Cargando y preparando datos...")
    secuencias, objetivos, max_esc = cargar_y_preparar_datos(archivo)
    print(f"🔢 Total de muestras generadas: {len(secuencias)}")

    tokenizer, X = tokenizar(secuencias, max_len=VENTANA_DIAS)
    y = objetivos

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Entrenando modelo Transformer...")
    start = time.time()
    model = transformer_encoder_model(len(tokenizer.word_index) + 1)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=16, verbose=1)

    print("📊 Evaluando modelo...")
    y_pred_norm = model.predict(X_test).flatten()
    y_pred = y_pred_norm * max_esc
    y_test_real = y_test * max_esc
    elapsed = time.time() - start

    mse = mean_squared_error(y_test_real, y_pred)
    mae = mean_absolute_error(y_test_real, y_pred)
    sse = np.sum((y_test_real - y_pred) ** 2)
    nse = 1 - (sse / np.sum((y_test_real - np.mean(y_test_real)) ** 2))

    graficar(y_test_real, y_pred, mse, mae, sse, nse, elapsed)

if __name__ == "__main__":
    main()
