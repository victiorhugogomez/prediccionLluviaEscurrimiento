import pandas as pd
import numpy as np

# Fijar la semilla para reproducibilidad
np.random.seed(42)

# Generar datos
data = {
    "ID": np.arange(1, 101),
    "Edad": np.random.randint(18, 70, size=100),  # Edad entre 18 y 70
    "Ingresos anuales": np.random.randint(20000, 120000, size=100),  # Ingresos entre 20,000 y 120,000
    "Gastos anuales": np.random.randint(10000, 80000, size=100),  # Gastos entre 10,000 y 80,000
    "Ahorros anuales": np.random.randint(5000, 50000, size=100),  # Ahorros entre 5,000 y 50,000
    "hors trabajadas anuales": np.random.randint(1200, 3000, size=100)  # Ahorros entre 5,000 y 50,000
}

# Crear DataFrame
df = pd.DataFrame(data)

# Guardar en un archivo CSV
df.to_csv("datos_financieros.csv", index=False)

print("Base de datos generada con éxito!")
print(df.head())  # Mostrar las primeras filas
