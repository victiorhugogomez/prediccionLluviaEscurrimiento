import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar el archivo CSV
data = pd.read_csv("C:/Users/dell/Documents/python/merged_data_cabra1.csv")
  # Asegúrate de ajustar el nombre del archivo

# Inspeccionar los datos
print(data.head())  # Muestra las primeras filas del dataset
print(data.info())  # Información sobre las columnas

# Convertir comas a puntos y asegurarse de que las columnas son numéricas
data['Streamflow'] = data['Streamflow'].str.replace(',', '.').astype(float)
data['p_ens'] = data['p_ens'].str.replace(',', '.').astype(float)

# Validar la conversión
print(data.head())

# Separar las características (X) y la etiqueta (y)
X = data[['Year', 'Month', 'Day', 'p_ens']].values  # Características
y = data['Streamflow'].values  # Etiqueta (lo que queremos predecir)

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de red neuronal
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1)  # Salida para regresión
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)

# Evaluar el modelo
loss, mae = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Mean Absolute Error: {mae}")

# Realizar predicciones
predictions = model.predict(X_test)
for i in range(10):  # Cambia 10 por el número de comparaciones que desees mostrar
    print(f"Predicción: {predictions[i][0]:.2f}, Valor real: {y_test[i]:.2f}")
