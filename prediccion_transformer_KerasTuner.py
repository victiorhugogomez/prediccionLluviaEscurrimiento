import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras_tuner as kt
import time
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# --------- Cargar y preparar los datos ----------
def select_file(title="Selecciona archivo Excel"):
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("Excel files", "*.xlsx")])

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

# ---------- Positional Encoding ----------
class PositionalEncodingLearnable(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pos_embedding = self.add_weight(name="pos_embed", shape=(1, seq_len, d_model))

    def call(self, x):
        return x + self.pos_embedding

# ---------- Submodelos pruneados ----------
def get_submodelo_config(modelo_id):
    if modelo_id == 0:
        return {"num_layers": 4, "dense_units": 256, "key_dim": 32}
    elif modelo_id == 1:
        return {"num_layers": 3, "dense_units": 128, "key_dim": 32}
    elif modelo_id == 2:
        return {"num_layers": 2, "dense_units": 128, "key_dim": 16}
    elif modelo_id == 3:
        return {"num_layers": 1, "dense_units": 64, "key_dim": 8}
    else:
        raise ValueError("modelo_id inválido")

# ---------- Métrica personalizada ----------
def calcular_metrica_personalizada(model, y_true, y_pred, base_model_params, alpha=0.7):
    mse = mean_squared_error(y_true, y_pred)
    sse = np.sum((y_true - y_pred) ** 2)
    nse = 1 - (sse / np.sum((y_true - np.mean(y_true)) ** 2))
    current_params = model.count_params()
    compression_rate = current_params / base_model_params
    custom_metric = alpha * nse + (1 - alpha) * (1 - compression_rate)
    return custom_metric

# ---------- Clase HyperModel para Keras Tuner ----------
class TransformerHyperModel(kt.HyperModel):
    def __init__(self, input_shape, y_train, y_test, X_train, X_test):
        self.input_shape = input_shape
        self.y_train = y_train
        self.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test

    def build(self, hp):
        modelo_id = hp.Int("modelo_id", 0, 3)
        config = get_submodelo_config(modelo_id)

        num_layers = config["num_layers"]
        dense_units = config["dense_units"]
        key_dim = config["key_dim"]
        num_heads = hp.Choice("num_heads", [2, 4, 8])
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.4)
        
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Dense(dense_units)(inputs)
        x = PositionalEncodingLearnable(self.input_shape[0], dense_units)(x)

        for _ in range(num_layers):
            attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
            x = layers.Add()([x, attn])
            x = layers.LayerNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.Choice("batch_size", [8, 16, 32])
        model.fit(self.X_train, self.y_train,
                  validation_data=(self.X_test, self.y_test),
                  batch_size=batch_size,
                  epochs=30,
                  verbose=0)

        y_pred = model.predict(self.X_test).flatten()
        base_model_params = 250000  # modelo base más grande
        return calcular_metrica_personalizada(model, self.y_test, y_pred, base_model_params)

# ---------- Entrenamiento final ----------
def main():
    file = select_file()
    if not file:
        print("No se seleccionó archivo.")
        return

    ventana_dias = 60
    X, y = cargar_datos(file, ventana_dias)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = kt.Hyperband(
        TransformerHyperModel(X.shape[1:], y_train, y_test, X_train, X_test),
        objective=kt.Objective("score", direction="max"),
        max_epochs=30,
        directory="ktuner_logs",
        project_name="transformer_autocompresion"
    )

    tuner.search(X_train, y_train)

    best_model = tuner.get_best_models(1)[0]
    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("🏆 Mejores hiperparámetros:")
    for k, v in best_hp.values.items():
        print(f"{k}: {v}")

    print("\n🎯 Evaluando modelo final...")
    start = time.time()
    y_pred = best_model.predict(X_test).flatten()
    elapsed = time.time() - start

    base_model_params = 250000
    metrica = calcular_metrica_personalizada(best_model, y_test, y_pred, base_model_params)

    plt.plot(y_test, label='Medido')
    plt.plot(y_pred, label='Predicho')
    plt.title(f'Modelo Keras Tuner\nMétrica personalizada: {metrica:.4f} - Tiempo: {elapsed:.2f}s')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
