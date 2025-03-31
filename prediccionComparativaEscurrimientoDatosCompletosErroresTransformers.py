import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D, Add
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tkinter import Tk, filedialog

# ---------- Positional Encoding (no entrenable) ----------
def add_position_encoding(X):
    pos = np.arange(X.shape[1])[:, np.newaxis]
    d_model = X.shape[2]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return X + pos_encoding

# ---------- Transformer Encoder Block ----------
def transformer_encoder_block(x, num_heads, ff_dim, dropout_rate):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(x.shape[-1])(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# ---------- Transformer Model ----------
def train_deep_transformer(X_train, y_train, num_heads=8, ff_dim=256, num_layers=2, dropout_rate=0.1):
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = input_layer

    # Encoder bloques
    for _ in range(num_layers):
        x = transformer_encoder_block(x, num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate)

    x = GlobalAveragePooling1D()(x)
    output_layer = Dense(1)(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    return model

# ---------- Resto del código ----------
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

def main():
    print("Selecciona el archivo de datos combinados")
    data_file = select_file("Selecciona el archivo de datos combinados")
    
    if not data_file:
        print("No se seleccionó ningún archivo. Terminando.")
        return
    
    df = pd.read_excel(data_file)
    df = df.dropna()

    # Normalización
    df_scaled = df.copy()
    for col in df.columns:
        df_scaled[col] = df[col] / df[col].max()

    look_back = 10

    # ---------- Con precipitación ----------
    X_prec, y_prec = prepare_data(df_scaled, ['p_ens'], 'Streamflow', look_back=look_back)
    X_prec = add_position_encoding(X_prec)
    model_prec = train_deep_transformer(X_prec, y_prec)
    y_pred_prec = model_prec.predict(X_prec)
    mse_prec, mae_prec, sse_prec, nse_prec = calculate_errors(y_prec, y_pred_prec)
    plot_results(y_prec, y_pred_prec, "Predicción con Precipitación", mse_prec, mae_prec, sse_prec, nse_prec)

    # ---------- Con todos los factores ----------
    feature_cols = ['p_ens', 'tmin_ens', 'tmax_ens', 'rh_ens', 'wnd_ens', 'srad_ens', 'et_ens', 'pet_pm', 'pet_pt', 'pet_hg']
    X_all, y_all = prepare_data(df_scaled, feature_cols, 'Streamflow', look_back=look_back)
    X_all = add_position_encoding(X_all)
    model_all = train_deep_transformer(X_all, y_all)
    y_pred_all = model_all.predict(X_all)
    mse_all, mae_all, sse_all, nse_all = calculate_errors(y_all, y_pred_all)
    plot_results(y_all, y_pred_all, "Predicción con Todos los Factores Meteorológicos", mse_all, mae_all, sse_all, nse_all)

if __name__ == "__main__":
    main()
