
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout
from tensorflow.keras.layers import MultiHeadAttention, GlobalAveragePooling1D, Add
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

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

def transformer_model(input_shape, heads=4, ff_dim=64, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attention_output = MultiHeadAttention(num_heads=heads, key_dim=input_shape[-1])(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)
    return Model(inputs=inputs, outputs=outputs)

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

    df_scaled = df.copy()
    for col in df.columns:
        df_scaled[col] = df[col] / df[col].max()

    # Solo precipitación
    X_prec, y_prec = prepare_data(df_scaled, ['p_ens'], 'Streamflow')
    model_prec = transformer_model(X_prec.shape[1:])
    model_prec.compile(optimizer='adam', loss='mse')
    model_prec.fit(X_prec, y_prec, epochs=50, batch_size=16, verbose=1)
    y_pred_prec = model_prec.predict(X_prec)
    mse, mae, sse, nse = calculate_errors(y_prec, y_pred_prec)
    plot_results(y_prec, y_pred_prec, "Predicción con Precipitación (Transformer)", mse, mae, sse, nse)

    # Todos los factores meteorológicos
    features = ['p_ens', 'tmin_ens', 'tmax_ens', 'rh_ens', 'wnd_ens', 'srad_ens', 'et_ens', 'pet_pm', 'pet_pt', 'pet_hg']
    X_all, y_all = prepare_data(df_scaled, features, 'Streamflow')
    model_all = transformer_model(X_all.shape[1:])
    model_all.compile(optimizer='adam', loss='mse')
    model_all.fit(X_all, y_all, epochs=50, batch_size=16, verbose=1)
    y_pred_all = model_all.predict(X_all)
    mse_all, mae_all, sse_all, nse_all = calculate_errors(y_all, y_pred_all)
    plot_results(y_all, y_pred_all, "Predicción con Todos los Factores (Transformer)", mse_all, mae_all, sse_all, nse_all)

if __name__ == "__main__":
    main()
