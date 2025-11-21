import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# VARIABLES GLOBALES
# -----------------------------
df = None
columnas_seleccionadas = []
varlist = {}
model = None
y_test = None
y_pred = None

# -----------------------------
# INTERFAZ PRINCIPAL
# -----------------------------
root = tk.Tk()
root.title("Sistema Predictivo Retail - Regresión Logística")

# -----------------------------
# LABEL DE ESTADO
# -----------------------------
label_estado = ttk.Label(root, text="Dataset no cargado")
label_estado.pack(pady=10)

# -----------------------------
# BOTON CARGAR DATASET
# -----------------------------
def cargar_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    df = pd.read_csv(file_path)
    mediana = df["IngresoTotal"].median()
    df["IngresoBinario"] = (df["IngresoTotal"] < mediana).astype(int)
    label_estado.config(text=f"Dataset cargado. Mediana IngresoTotal: {mediana:.2f}")
    messagebox.showinfo("OK", "Dataset cargado y variable IngresoBinario creada.")

ttk.Button(root, text="Cargar Dataset", command=cargar_dataset).pack(pady=10)

# -----------------------------
# BOTON CONFIGURAR MODELO
# -----------------------------
def configurar_modelo():
    if df is None:
        messagebox.showwarning("Error", "Primero debe cargar el dataset.")
        return
    
    win = tk.Toplevel(root)
    win.title("Seleccionar columnas")
    
    ttk.Label(win, text="Seleccione columnas para usar como predictores:").pack(pady=5)
    frame = ttk.Frame(win)
    frame.pack()
    
    global varlist
    varlist = {}
    
    for col in df.columns:
        if col != "IngresoBinario":
            var = tk.BooleanVar()
            chk = ttk.Checkbutton(frame, text=col, variable=var)
            chk.pack(anchor="w")
            varlist[col] = var
    
    def guardar_columnas():
        global columnas_seleccionadas
        columnas_seleccionadas = [col for col, var in varlist.items() if var.get()]
        if "IngresoTotal" not in columnas_seleccionadas:
            columnas_seleccionadas.append("IngresoTotal")
        messagebox.showinfo("OK", f"Columnas seleccionadas:\n{columnas_seleccionadas}")
    
    ttk.Button(win, text="Guardar selección", command=guardar_columnas).pack(pady=10)

ttk.Button(root, text="Configurar Modelo", command=configurar_modelo).pack(pady=10)

# -----------------------------
# BOTON ENTRENAR MODELO
# -----------------------------
def entrenar_modelo():
    global model, y_test, y_pred
    if df is None:
        messagebox.showwarning("Error", "Debe cargar el dataset primero.")
        return
    if not columnas_seleccionadas:
        messagebox.showwarning("Error", "Debe seleccionar columnas.")
        return
    
    X = df[columnas_seleccionadas]
    y = df["IngresoBinario"]
    X = pd.get_dummies(X, drop_first=True)
    
    X_train, X_test, y_train, y_test_local = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred_local = model.predict(X_test)
    
    y_test = y_test_local
    y_pred = y_pred_local
    
    messagebox.showinfo("OK", "Modelo entrenado correctamente.")

ttk.Button(root, text="Entrenar Modelo", command=entrenar_modelo).pack(pady=10)

# -----------------------------
# BOTON RESULTADOS
# -----------------------------
def mostrar_resultados():
    if model is None:
        messagebox.showwarning("Error", "Primero debe entrenar el modelo.")
        return
    
    win = tk.Toplevel(root)
    win.title("Resultados del Modelo")
    
    text = tk.Text(win, width=80, height=20)
    text.pack()
    
    text.insert(tk.END, "=== INFORME REGRESIÓN LOGÍSTICA ===\n\n")
    text.insert(tk.END, classification_report(y_test, y_pred))
    
    def mostrar_matriz():
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.show()
    
    ttk.Button(win, text="Mostrar Matriz de Confusión", command=mostrar_matriz).pack(pady=10)

ttk.Button(root, text="Mostrar Resultados", command=mostrar_resultados).pack(pady=10)

# -----------------------------
# INSTRUCCIONES / EXPLICACIÓN
# -----------------------------
explicacion = """INSTRUCCIONES:
1. Cargar dataset CSV (el programa creará IngresoBinario automáticamente).
2. Seleccionar columnas predictoras (IngresoTotal será obligatorio).
3. Entrenar modelo con regresión logística.
4. Mostrar resultados y matriz de confusión.
5. Los gráficos se mostrarán en ventanas separadas.
"""
ttk.Label(root, text=explicacion, justify="left").pack(pady=10)

root.mainloop()
