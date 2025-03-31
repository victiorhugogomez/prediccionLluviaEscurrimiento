import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV
file_path = 'datosSep.csv'
data = pd.read_csv(file_path)

# Seleccionar las columnas de interés
clean_data = data[['Solar', 'ET']]

# Verificar valores nulos en Solar
print("Valores no válidos en Solar:", clean_data[clean_data['Solar'].isna()])

# Reemplazar o eliminar valores nulos en la columna Solar
clean_data = clean_data.dropna(subset=['Solar'])

# Asegurar que las columnas sean numéricas usando .loc[]
clean_data.loc[:, 'Solar'] = pd.to_numeric(clean_data['Solar'], errors='coerce')
clean_data.loc[:, 'ET'] = pd.to_numeric(clean_data['ET'], errors='coerce')

# Eliminar filas con valores nulos restantes
clean_data = clean_data.dropna()

# Asegurarse de que no se hayan eliminado todos los datos
if clean_data.empty:
    raise ValueError("No hay datos suficientes después de eliminar los valores nulos o no numéricos.")

# Definir la variable independiente (Solar) y dependiente (ET)
X = clean_data['Solar'].values.reshape(-1, 1)  # Variable independiente
y = clean_data['ET'].values  # Variable dependiente

# Crear el modelo de regresión lineal
model = LinearRegression()

# Ajustar el modelo
model.fit(X, y)

# Obtener el coeficiente y la intersección
coef = model.coef_[0]
intercept = model.intercept_

# Predecir los valores de evapotranspiración
y_pred = model.predict(X)

# Agregar los valores predichos al DataFrame original
clean_data['Predicted_ET'] = y_pred

# Guardar los datos en un nuevo archivo CSV o Excel
clean_data.to_csv('nuevo_archivo.csv', index=False)

# Crear la gráfica de dispersión con la línea de regresión
plt.figure(figsize=(8, 6))
plt.scatter(clean_data['Solar'], clean_data['ET'], color='blue', label='Datos reales')
plt.plot(clean_data['Solar'], y_pred, color='red', label='Regresión lineal')
plt.xlabel('Radiación Solar')
plt.ylabel('Evapotranspiración')
plt.title('Regresión Lineal: Radiación Solar vs Evapotranspiración')
plt.legend()
plt.grid(True)
plt.show()
