import pandas as pd
from tkinter import Tk, filedialog

def select_file(title):
    root = Tk()
    root.withdraw()  # Ocultar la ventana de Tkinter
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Excel Files", "*.xlsx")])
    return file_path

def merge_files():
    print("Selecciona el archivo de escurrimiento")
    streamflow_file = select_file("Selecciona el archivo de escurrimiento")
    
    print("Selecciona el archivo de datos meteorológicos")
    climate_file = select_file("Selecciona el archivo de datos meteorológicos")
    
    if not streamflow_file or not climate_file:
        print("No se seleccionaron ambos archivos. Terminando.")
        return
    
    # Leer los archivos
    streamflow_sheets = pd.ExcelFile(streamflow_file).sheet_names
    climate_sheets = pd.ExcelFile(climate_file).sheet_names
    
    # Cargar los datos de la primera hoja
    streamflow_df = pd.read_excel(streamflow_file, sheet_name=streamflow_sheets[0])
    climate_df = pd.read_excel(climate_file, sheet_name=climate_sheets[0])
    
    # Limpiar nombres de columnas eliminando espacios extra
    streamflow_df.columns = streamflow_df.columns.str.strip()
    climate_df.columns = climate_df.columns.str.strip()
    
    # Unir los datos en base a Year, Month y Day
    merged_df = pd.merge(streamflow_df, climate_df, on=["Year", "Month", "Day"], how="inner")
    
    # Guardar el archivo combinado
    save_path = filedialog.asksaveasfilename(
        defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")], title="Guardar archivo combinado"
    )
    
    if save_path:
        merged_df.to_excel(save_path, index=False)
        print(f"Archivo guardado exitosamente en: {save_path}")
    else:
        print("No se guardó el archivo.")

if __name__ == "__main__":
    merge_files()
