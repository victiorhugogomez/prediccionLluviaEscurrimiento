import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

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

def plot_results(y_true, y_pred, title):
    plt.figure(figsize=(10,5))
    plt.plot(y_true, label='Medido', color='blue')
    plt.plot(y_pred, label='Predicho', color='red', linestyle='dashed')
    plt.xlabel('Tiempo')
    plt.ylabel('Escurrimiento')
    plt.legend()
    plt.title(title)
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
    
    # Predicción con solo precipitación
    X_prec, y_prec = prepare_data(df_scaled, ['p_ens'], 'Streamflow')
    model_prec = train_lstm(X_prec, y_prec)
    y_pred_prec = model_prec.predict(X_prec)
    plot_results(y_prec, y_pred_prec, "Predicción con Precipitación")
    
    # Predicción con todos los factores meteorológicos
    feature_cols = ['p_ens', 'tmin_ens', 'tmax_ens', 'rh_ens', 'wnd_ens', 'srad_ens', 'et_ens', 'pet_pm', 'pet_pt', 'pet_hg']
    X_all, y_all = prepare_data(df_scaled, feature_cols, 'Streamflow')
    model_all = train_lstm(X_all, y_all)
    y_pred_all = model_all.predict(X_all)
    plot_results(y_all, y_pred_all, "Predicción con Todos los Factores Meteorológicos")
    
if __name__ == "__main__":
    main()
