import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd

#Variablkes gobales : Es mejor que estar llamando funciones a cada rato
dataset = None
mediana = None

# Iniciar interfas de tkinter
root = tk.Tk()
root.title("Sistema Predictivo Retail - Regresión Logística")
root.geometry("600x400")

# Funciones para los botones
def cargar_dataset():
    global dataset
    archivo_ubicacion = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not archivo_ubicacion:
        return
    dataset = pd.read_csv(archivo_ubicacion)
    messagebox.showinfo("OK", "Dataset cargado y variable IngresoBinario creada.")

def convertir_a_binario():
    global dataset, mediana
    if dataset is None:
        messagebox.showwarning("Error", "Primero debe cargar el dataset.")
        return
    mediana = dataset["IngresoTotal"].median()
    dataset["IngresoBinario"] = (dataset["IngresoTotal"] < mediana).astype(int)
    messagebox.showinfo("OK", f"Variable IngresoBinario creada con mediana: {mediana:.2f}")


#Botones

ttk.Button(root, text="Cargar Dataset", command=cargar_dataset).pack(pady=10)
ttk.Button(root, text="print xd", command=convertir_a_binario).pack(pady=10)

#Loop que mantiene la ventana abierta : NO ELIMINAR
root.mainloop()