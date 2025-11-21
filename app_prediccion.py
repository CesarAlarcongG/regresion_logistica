
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PredictApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predicción Retail - Regresión Logística")
        self.root.geometry("900x650")
        self.root.configure(bg="#F5F5F5")

        self.data = None
        self.encoded_data = None
        self.columns_list = []

        title = tk.Label(root, text="PREDICCIÓN DE INGRESOS BAJOS/ALTOS",
                         font=("Arial", 18, "bold"), bg="#F5F5F5")
        title.pack(pady=10)

        # --- CARGA DE ARCHIVO ---
        frame_file = tk.Frame(root, bg="#F5F5F5")
        frame_file.pack(pady=10)

        btn_load = tk.Button(frame_file, text="Cargar CSV", command=self.load_file)
        btn_load.grid(row=0, column=0, padx=10)

        self.label_file = tk.Label(frame_file, text="Archivo no cargado", bg="#F5F5F5")
        self.label_file.grid(row=0, column=1)

        # --- LISTA DE COLUMNAS ---
        self.frame_columns = tk.Frame(root, bg="#F5F5F5")
        self.frame_columns.pack(pady=10)

        tk.Label(root, text="Seleccione columnas predictoras:", bg="#F5F5F5").pack()

        self.listbox = tk.Listbox(self.frame_columns, selectmode=tk.MULTIPLE, width=50)
        self.listbox.pack(side="left")

        scrollbar = tk.Scrollbar(self.frame_columns, orient="vertical")
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")

        self.listbox.config(yscrollcommand=scrollbar.set)

        # Botón entrenar
        btn_train = tk.Button(root, text="Entrenar Modelo", command=self.train_model,
                              bg="#D1E7DD", width=20)
        btn_train.pack(pady=10)

        # Cuadro de resultados
        self.text_results = tk.Text(root, width=100, height=17)
        self.text_results.pack(pady=10)

    # ----------------------------------------------------------------
    def load_file(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Seleccionar archivo CSV",
                filetypes=[("CSV files", "*.csv")]
            )
            if not file_path:
                return

            self.data = pd.read_csv(file_path)
            self.label_file.config(text=f"Cargado: {file_path.split('/')[-1]}")

            # --- PROCESAR COLUMNAS CATEGÓRICAS ---
            self.encoded_data = pd.get_dummies(self.data, columns=["Zona", "DiaSemana"], drop_first=True)

            self.columns_list = [c for c in self.encoded_data.columns if c != "IngresoTotal"]

            self.listbox.delete(0, tk.END)
            for col in self.columns_list:
                self.listbox.insert(tk.END, col)

            messagebox.showinfo("Éxito", "Archivo cargado correctamente.")

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar archivo:\n{e}")

    # ----------------------------------------------------------------
    def train_model(self):
        if self.encoded_data is None:
            messagebox.showwarning("Advertencia", "Cargue un archivo primero.")
            return

        try:
            # Convertir IngresoTotal a binario
            mediana = self.encoded_data["IngresoTotal"].median()
            self.encoded_data["IngresoBin"] = self.encoded_data["IngresoTotal"].apply(
                lambda x: 1 if x < mediana else 0
            )
        except:
            messagebox.showerror("Error", "La columna 'IngresoTotal' no existe.")
            return

        # Leer columnas seleccionadas
        selected_indices = self.listbox.curselection()
        selected_cols = [self.listbox.get(i) for i in selected_indices]

        if len(selected_cols) == 0:
            messagebox.showwarning("Advertencia", "Debe seleccionar al menos una variable predictora.")
            return

        # Variables para el modelo
        X = self.encoded_data[selected_cols]
        y = self.encoded_data["IngresoBin"]

        # Normalización de variables numéricas
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )

            model = LogisticRegression(max_iter=1500)
            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            cm = confusion_matrix(y_test, pred)
            cr = classification_report(y_test, pred)

            # Mostrar resultados
            self.text_results.delete("1.0", tk.END)
            self.text_results.insert(tk.END, "=== INFORME DEL MODELO ===\n\n")
            self.text_results.insert(tk.END, f"Columnas usadas: {selected_cols}\n\n")
            self.text_results.insert(tk.END, f"Mediana de IngresoTotal: {mediana:.2f}\n")
            self.text_results.insert(tk.END, "IngresoBin = 1 si ingreso < mediana (Bajo)\nIngresoBin = 0 si ingreso >= mediana (Alto)\n\n")
            self.text_results.insert(tk.END, f"Exactitud del modelo: {acc:.4f}\n\n")
            self.text_results.insert(tk.END, "Matriz de Confusión:\n")
            self.text_results.insert(tk.END, f"{cm}\n\n")
            self.text_results.insert(tk.END, "Reporte de Clasificación:\n")
            self.text_results.insert(tk.END, cr)

            self.show_confusion_matrix(cm)

        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento:\n{e}")

    # ----------------------------------------------------------------
    def show_confusion_matrix(self, cm):
        win = tk.Toplevel(self.root)
        win.title("Matriz de Confusión")
        win.geometry("500x450")

        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Matriz de Confusión")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack()


# ----------------------------------------------------------------
root = tk.Tk()
app = PredictApp(root)
root.mainloop()
