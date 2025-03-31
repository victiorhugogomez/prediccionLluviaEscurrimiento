import pandas as pd
import numpy as np
# import ace_tools
# import ace_tools as tools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import os

file_paths = [
    'C:/Users/dell/Documents/Maestria/python/merged_data_cabra1.xlsx',
    'C:/Users/dell/Documents/Maestria/python/merged_data_cabra2.xlsx',
    'C:/Users/dell/Documents/Maestria/python/merged_data_cabra3.xlsx',
    'C:/Users/dell/Documents/Maestria/python/merged_data_cabra5.xlsx'
]
# === Función para normalizar datos respecto a su máximo ===
def normalize_data(df):
    """
    Normaliza las columnas de escurrimiento y precipitación dividiéndolas por su valor máximo.
    """
    df["Streamflow_norm"] = df["Streamflow"] / df["Streamflow"].max()
    df["p_ens_norm"] = df["p_ens"] / df["p_ens"].max()
    return df

# === Función 1: Cargar y limpiar datos ===
def load_and_clean_data(file_paths):
    """
    Carga múltiples archivos Excel y realiza limpieza básica:
    - Combina datos de varios archivos.
    - Crea una columna de fecha combinada.
    """
    dfs = []
    for file_path in file_paths:
        df = pd.read_excel(file_path)
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        df = df[["Date", "Streamflow", "p_ens"]]
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.dropna(inplace=True)  # Eliminar filas con valores faltantes
    return combined_df

# === Función 2: Análisis exploratorio ===
def exploratory_analysis(df):
    """
    Genera gráficos iniciales para analizar las series temporales de precipitación y escurrimiento.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["p_ens_norm"], label="Precipitación (p_ens)", color="blue")
    plt.plot(df["Date"], df["Streamflow_norm"], label="Escurrimiento (Streamflow)", color="green")
    plt.xlabel("Fecha")
    plt.ylabel("Valores")
    plt.title("Precipitación y Escurrimiento")
    plt.legend()
    plt.grid()
    plt.show()

    # Gráfico de correlación entre variables
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.show()

# === Función 3: Calcular desfase ===
def calculate_lag(df):
    """
    Calcula el desfase óptimo entre precipitación y escurrimiento usando correlación cruzada.
    """
    precip = df["p_ens_norm"].values
    streamflow = df["Streamflow_norm"].values
    lags = np.arange(-len(precip) + 1, len(precip))
    correlation = np.correlate(precip - np.mean(precip), streamflow - np.mean(streamflow), mode="full")
    optimal_lag = lags[np.argmax(correlation)]
    return optimal_lag

# === Función 4: Ajustar datos según desfase ===
def adjust_data_by_lag(df, lag):
    """
    Ajusta los datos de precipitación con el desfase calculado.
    """
    df["p_ens_lagged"] = df["p_ens_norm"].shift(lag)
    df.dropna(inplace=True)  # Eliminar valores NaN introducidos por el desfase
    return df

# === Función 5: Entrenar modelo ===
def train_model(df):
    """
    Entrena una red neuronal simple para predecir el escurrimiento.
    """
    X = df[["p_ens_lagged"]].values
    y = df["Streamflow_norm"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluar el modelo
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}, MAE: {mae}")

    return model, scaler

# === Función 6: Visualizar resultados ===
def plot_results(df, model, scaler):
    """
    Genera predicciones y muestra un gráfico comparativo entre valores reales y predichos.
    """
    X = df[["p_ens_lagged"]].values
    y = df["Streamflow_norm"].values

    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], y, label="Valores Reales", color="green")
    plt.plot(df["Date"], predictions, label="Predicciones", color="red")
    plt.xlabel("Fecha")
    plt.ylabel("Escurrimiento")
    plt.title("Comparación de Escurrimiento Real vs Predicho")
    plt.legend()
    plt.grid()
    plt.show()

# === Flujo principal ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_paths = [
            os.path.join(script_dir, "merged_data_cabra1.xlsx"),
            os.path.join(script_dir, "merged_data_cabra2.xlsx"),
            os.path.join(script_dir, "merged_data_cabra3.xlsx"),
            os.path.join(script_dir, "merged_data_cabra5.xlsx"),
    ]  
    # file_paths = [
    #     'C:/Users/dell/Documents/Maestria/python/merged_data_cabra1.xlsx',
    #     'C:/Users/dell/Documents/Maestria/python/merged_data_cabra2.xlsx',
    #     'C:/Users/dell/Documents/Maestria/python/merged_data_cabra3.xlsx',
    #     'C:/Users/dell/Documents/Maestria/python/merged_data_cabra5.xlsx'
    # ]

    # Cargar y limpiar datos
    df = load_and_clean_data(file_paths)
    # Normalizar los datos
    df = normalize_data(df)
    print(df.head(100))
    df.head(100).to_csv("datos_mostrados.csv", index=False)
    print("Archivo guardado como datos_mostrados.csv, ábrelo en Excel para verlo.")
    # ace_tools.display_dataframe_to_user(name="Primeros 100 registros", dataframe=df.head(100))

    # Análisis exploratorio
    exploratory_analysis(df)

    # Calcular y ajustar desfase
    lag = calculate_lag(df)
    adjusted_df = adjust_data_by_lag(df, lag)
    print(f"Desfase óptimo calculado: {lag} días.")

    # # Entrenar modelo y visualizar resultados
    # model, scaler = train_model(adjusted_df)
    # plot_results(adjusted_df, model, scaler)
