
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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

from pyswarm import pso

def objective_function(params, X, y):
    units, batch_size, epochs = map(int, params)
    model = Sequential([
        LSTM(units, activation='relu', return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        LSTM(units, activation='relu'), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred)

def optimize(X, y):
    lb = [10, 8, 10]
    ub = [100, 64, 100]
    best_params, _ = pso(lambda p: objective_function(p, X, y), lb, ub, swarmsize=10, maxiter=5)
    return list(map(int, best_params))
