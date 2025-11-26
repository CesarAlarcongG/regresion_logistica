
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, colorchooser
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import os

# --- Archivo: app_prediccion.py ---
# Descripción: Aplicación GUI para predicción de ingresos Altos y Bajos usando Regresión Logística.

class PredictApp:
    def __init__(self, root):
        """Inicializa la interfaz gráfica y las variables de estado."""
        self.root = root
        self.root.title("Sistema de Predicción Retail - Regresión Logística")
        self.root.geometry("1280x720")
        self.root.state("zoomed")
        self.root.configure(bg="#F0F8FF")

        # Variables de estado para datos y modelo
        self.data = None            # DataFrame original
        self.encoded_data = None    # DataFrame procesado (One-Hot Encoding)
        self.columns_list = []      # Lista de columnas predictoras disponibles

        self.last_file_path = None
        self.model = None
        self.scaler = None
        self.last_cm = None         # Última matriz de confusión generada
        self.last_results_df = None # DataFrame con predicciones para exportar
        self.umbral = None        # Umbral de clasificación
        # --- Encabezado ---
        title = tk.Label(root, text="PREDICCIÓN INGRESOS BAJOS/ALTOS (Regresión Logística)",
                         font=("Segoe UI", 16, "bold"), bg="#F0F8FF", fg="#0b5394")
        title.pack(pady=8)

        # --- Panel Superior: Carga de archivos y configuración ---
        top_frame = tk.Frame(root, bg="#F0F8FF")
        top_frame.pack(fill="x", padx=12)

        btn_load = tk.Button(top_frame, text="📂 Cargar CSV", command=self.load_file, bg="#4caf50", fg="white")
        btn_load.grid(row=0, column=0, padx=6, pady=6)

        self.label_file = tk.Label(top_frame, text="Archivo no cargado", bg="#F0F8FF")
        self.label_file.grid(row=0, column=1, padx=6)

        btn_graphs = tk.Button(top_frame, text="📈 Abrir Ventana de Gráficos", command=self.open_graph_window)
        btn_graphs.grid(row=0, column=3, padx=6)

        # Opción para limitar columnas (útil en datasets muy anchos)
        tk.Label(top_frame, text="Limitar columnas a usar (0=usar todas):", bg="#F0F8FF").grid(row=1, column=0, sticky="e", padx=6)
        self.limit_var = tk.IntVar(value=0)
        tk.Entry(top_frame, textvariable=self.limit_var, width=6).grid(row=1, column=1, sticky="w")

        # Selector de color para gráficos
        tk.Label(top_frame, text="Color gráficos:", bg="#F0F8FF").grid(row=1, column=2, sticky="e")
        self.color_btn = tk.Button(top_frame, text="Elegir color", command=self.choose_color)
        self.color_btn.grid(row=1, column=3, sticky="w")
        self.plot_color = "#0b5394"  # Color por defecto

        # --- Panel Central: Selección de variables y Consola ---
        middle_frame = tk.Frame(root, bg="#F0F8FF")
        middle_frame.pack(fill="both", expand=False, padx=12, pady=6)

        # Columna Izquierda: Lista de variables
        left_col = tk.Frame(middle_frame, bg="#F0F8FF")
        left_col.pack(side="left", fill="y", padx=6)

        tk.Label(left_col, text="Seleccione columnas predictoras:", bg="#F0F8FF").pack(anchor="w")

        self.listbox = tk.Listbox(left_col, selectmode=tk.MULTIPLE, width=50, height=15)
        self.listbox.pack(side="left", fill="y")

        scrollbar = tk.Scrollbar(left_col, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        # Botones de control de lista
        btns_col = tk.Frame(middle_frame, bg="#F0F8FF")
        btns_col.pack(side="left", padx=12)

        tk.Button(btns_col, text="Mapa de correlacion", command=self.mapa_correlacion, width=28).pack(pady=8)
        tk.Button(btns_col, text="Seleccionar Todo", command=self.select_all, width=28).pack(pady=4)
        tk.Button(btns_col, text="Limpiar Selección", command=self.clear_selection, width=28).pack(pady=4)
        tk.Button(btns_col, text="Seleccionar N Aleatorio", command=self.select_random_n, width=28).pack(pady=4)
        tk.Button(btns_col, text="Exportar columnas detectadas", command=self.export_columns, width=28).pack(pady=4)

        # Columna Derecha: Resultados y Botón de Entrenamiento
        right_col = tk.Frame(middle_frame, bg="#F0F8FF")
        right_col.pack(side="left", fill="both", expand=True, padx=6)

        tk.Button(right_col, text="▶ Entrenar Modelo", command=self.train_model, bg="#1976d2", fg="white", width=20).pack(pady=6)

        self.text_results = tk.Text(right_col, width=85, height=25)
        self.text_results.pack(padx=6, pady=6)

        # --- Panel Inferior: Acciones finales ---
        bottom_frame = tk.Frame(root, bg="#F0F8FF")
        bottom_frame.pack(fill="x", padx=12, pady=6)

        tk.Button(bottom_frame, text="Guardar resultados (CSV)", command=self.save_results_csv).pack(side="left", padx=6)
        tk.Button(bottom_frame, text="Limpiar Consola", command=lambda: self.text_results.delete("1.0", tk.END)).pack(side="left", padx=6)

        # Barra de estado
        self.status_var = tk.StringVar(value="Esperando carga de archivo...")
        status = tk.Label(root, textvariable=self.status_var, anchor="w", bg="#d9ecff")
        status.pack(fill="x", padx=12, pady=(0,6))

    # --------------------------- Utilidades ---------------------------
    # ------------------------ Colores y paletas -----------------------
    def choose_color(self):
        """Abre diálogo y guarda color + paletas derivadas."""
        color = colorchooser.askcolor(title="Elige color principal para gráficos")
        if color and color[1]:
            self.plot_color = color[1]
            # generar paletas rápidas (6 tonos para pie por ejemplo)
            self._pie_shades = self.generate_shades(self.plot_color, 6)
            # cmap para heatmap
            self._heatmap_cmap = self.generate_diverging_cmap(self.plot_color)
            # si quieres forzar repintado de canvas actual:
            try:
                self.canvas.draw()
            except:
                pass

    def hex_to_rgb(self, hex_color):
        """Devuelve tupla (r,g,b) en rango 0..1 para matplotlib."""
        return mcolors.to_rgb(hex_color)

    def rgb_to_hex(self, rgb):
        """Convierte (r,g,b) 0..1 a '#rrggbb'."""
        return mcolors.to_hex(rgb)

    def generate_shades(self, base_hex, n):
        """
        Genera n tonos (del más oscuro al más claro) basados en base_hex.
        Retorna lista de hex.
        """
        base_rgb = mcolors.to_rgb(base_hex)
        # mezclamos entre base y blanco para producir tonos
        shades = []
        for i in range(n):
            t = i / max(n-1, 1)  # 0..1
            # hacemos que t=0 -> base, t=1 -> más claro (hacia blanco)
            mixed = tuple((1 - t) * c + t * 1.0 for c in base_rgb)
            shades.append(mcolors.to_hex(mixed))
        return shades

    def generate_diverging_cmap(self, base_hex, n=256):
        """
        Crea un cmap diverging/fill para heatmap partiendo del color base.
        Se genera un gradiente desde color más oscuro hasta color más claro.
        """
        base_rgb = mcolors.to_rgb(base_hex)
        # crear color oscuro (mezclado con negro) y color claro (mezclado con blanco)
        dark = tuple(c * 0.3 for c in base_rgb)
        light = tuple((c + 1.0)/2.0 for c in base_rgb)  # hacia blanco
        return LinearSegmentedColormap.from_list("customcmap", [dark, base_rgb, light], N=n)

    def load_file(self):
        """Carga datos desde CSV o Excel, limpia texto y aplica One-Hot Encoding."""
        try:
            file_path = filedialog.askopenfilename(title="Seleccionar archivo", 
                                                   filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls")])
            if not file_path:
                return

            # Lectura flexible (CSV o Excel)
            if file_path.lower().endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
            else:
                self.data = pd.read_csv(file_path)

            self.last_file_path = file_path
            self.label_file.config(text=f"Cargado: {os.path.basename(file_path)}")
            self.status_var.set("Limpiando y codificando categorías...")

            # Limpieza: Eliminar espacios y capitalizar texto en columnas clave
            for col in ["DiaSemana", "Zona"]:
                if col in self.data.columns:
                    self.data[col] = self.data[col].astype(str).str.strip().str.title()

            # One-Hot Encoding: Convierte variables categóricas en numéricas (dummies)
            cat_cols = [c for c in ["Zona", "DiaSemana"] if c in self.data.columns]
            if cat_cols:
                self.encoded_data = pd.get_dummies(self.data, columns=cat_cols, drop_first=False)
            else:
                self.encoded_data = self.data.copy()

            # Validación: Verificar existencia de variable objetivo
            if "IngresoTotal" not in self.encoded_data.columns:
                messagebox.showerror("Error", "La columna 'IngresoTotal' no fue encontrada. Verifique el nombre.")
                self.status_var.set("Error: IngresoTotal no encontrado")
                return

            # Preparar lista para el ListBox (excluyendo la variable objetivo)
            self.columns_list = [c for c in self.encoded_data.columns if c != "IngresoTotal"]
            self.listbox.delete(0, tk.END)
            for col in self.columns_list:
                self.listbox.insert(tk.END, col)
            
            # Halla el umbral óptimo (mediana) para clasificación
            self.umbral = self.encoded_data["IngresoTotal"].median()

            # Agrega un nuevo campo binario basado en el IngresoTotal
            self.encoded_data["IngresoBinario"] = self.encoded_data["IngresoTotal"].apply(lambda x: 1 if x < self.umbral else 0)

            self.status_var.set(f"Archivo cargado: {os.path.basename(file_path)} - {len(self.encoded_data)} filas")
            messagebox.showinfo("Éxito", "Carga completa. Seleccione variables de la lista para continuar.")
        
        except Exception as e:
            messagebox.showerror("Error al cargar", str(e))
            self.status_var.set("Error al cargar archivo")

    def mapa_correlacion(self):
        """Muestra un mapa de calor de correlación centrado, horizontal y con scrollbars."""

        if self.encoded_data is None:
            messagebox.showwarning("Advertencia", "Cargue un archivo primero.")
            return

        numeric_data = self.encoded_data.select_dtypes(include=np.number)
        if numeric_data.empty:
            messagebox.showwarning("Advertencia", "No hay columnas numéricas para calcular correlación.")
            return

        import seaborn as sns

        corr = numeric_data.corr()

        # ------------------- VENTANA FULLSCREEN -------------------
        win = tk.Toplevel(self.root)
        win.title("Mapa de Correlación")
        win.state("zoomed")
        win.resizable(False, False)
        win.configure(bg="#f7fbff")

        # Tamaño pantalla
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()

        # ------------------- SCROLL ÁREA -------------------
        container = tk.Frame(win, bg="#f7fbff")
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg="#f7fbff", highlightthickness=0)
        canvas.pack(side="left", fill="both", expand=True)

        v_scroll = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        h_scroll = tk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")

        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # frame interno
        frame_plot = tk.Frame(canvas, bg="#f7fbff")
        canvas_window = canvas.create_window((0, 0), window=frame_plot, anchor="n")

        # region scroll
        def update_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # centrar horizontalmente
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())

        frame_plot.bind("<Configure>", update_scroll)
        canvas.bind("<Configure>", update_scroll)

        # ---------------- FIGURA ----------------
        n = corr.shape[0]

        # NUEVO: figura más horizontal y menos alta
        figsize_x = max(10, min(40, n * 0.9))   # ancho incrementado
        figsize_y = max(6, min(20, n * 0.55))   # alto moderado

        fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

        cmap = getattr(self, "_heatmap_cmap", None)
        if cmap is None:
            cmap = self.generate_diverging_cmap(self.plot_color)

        sns.set(font_scale=0.8)

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            cbar=True,
            ax=ax,
            annot_kws={"size": 7}
        )

        ax.tick_params(axis="both", labelsize=8)
        ax.set_title("Mapa de Calor de Correlación", fontsize=14, pad=20)

        fig.tight_layout(pad=2)

        # colocar figura dentro del frame
        fig_canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(pady=20)   # centrado vertical con margen

    def select_all(self):
        self.listbox.select_set(0, tk.END)

    def clear_selection(self):
        self.listbox.select_clear(0, tk.END)

    def select_random_n(self):
        """Selecciona N columnas aleatorias de la lista para pruebas rápidas."""
        if not self.columns_list:
            return
        n = simpledialog.askinteger("Seleccionar N", "¿Cuántas columnas seleccionar aleatoriamente?", minvalue=1, maxvalue=len(self.columns_list))
        if not n:
            return
        import random
        chosen = random.sample(range(len(self.columns_list)), n)
        self.clear_selection()
        for i in chosen:
            self.listbox.select_set(i, i)

    def export_columns(self):
        """Guarda los nombres de las columnas detectadas en un archivo de texto."""
        if not self.columns_list:
            messagebox.showwarning("Advertencia", "No hay columnas para exportar.")
            return
        save = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
        if save:
            with open(save, "w", encoding="utf-8") as f:
                f.write("\n".join(self.columns_list))
            messagebox.showinfo("Exportado", f"Columnas exportadas a: {save}")

    # --------------------------- Modelo y Métricas ---------------------------
    def train_model(self):
        """
        Ejecuta el flujo completo de ML:
        1. Binarización del objetivo.
        2. Escalado de características.
        3. División Train/Test.
        4. Entrenamiento de Regresión Logística.
        5. Cálculo de Logits y Probabilidades.
        6. Reporte de resultados.
        """
        if self.encoded_data is None:
            messagebox.showwarning("Advertencia", "Cargue un archivo primero.")
            return

        try:
            # 1. Crear Variable Objetivo Binaria (Target)
            # Usamos la mediana para dividir en clases equilibradas (1: Bajo Ingreso, 0: Alto Ingreso)
            mediana = self.encoded_data["IngresoTotal"].median()
            self.encoded_data["IngresoBin"] = self.encoded_data["IngresoTotal"].apply(lambda x: 1 if x < mediana else 0)
        except Exception as e:
            messagebox.showerror("Error", f"No se puede procesar IngresoTotal: {e}")
            return

        # Obtener columnas seleccionadas por el usuario
        selected_indices = self.listbox.curselection()
        selected_cols = [self.listbox.get(i) for i in selected_indices]

        # Lógica de respaldo: Si no hay selección, usar límite numérico o todas
        if len(selected_cols) == 0:
            limit = int(self.limit_var.get()) if self.limit_var.get() is not None else 0
            if limit > 0:
                selected_cols = self.columns_list[:limit]
            else:
                messagebox.showwarning("Advertencia", "Debe seleccionar al menos una variable predictora.")
                return

        # Truncar columnas si el usuario estableció un límite explícito
        limit = int(self.limit_var.get()) if self.limit_var.get() is not None else 0
        if limit > 0 and limit < len(selected_cols):
            selected_cols = selected_cols[:limit]

        X = self.encoded_data[selected_cols].copy()
        y = self.encoded_data["IngresoBin"].copy()

        # Validación: Asegurar que X sea numérico. Si no, intentar convertir o descartar.
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        if non_numeric:
            try:
                X[non_numeric] = X[non_numeric].apply(pd.to_numeric)
            except:
                messagebox.showwarning("Advertencia", f"Se eliminarán columnas no numéricas: {non_numeric}")
                X = X.drop(columns=non_numeric)

        # 2. Normalización (StandardScaler)
        # Importante para que los coeficientes sean comparables y el algoritmo converja mejor.
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        try:
            # 3. División Train/Test (70% entrenamiento, 30% prueba)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None)
            
            # 4. Entrenamiento del Modelo
            self.model = LogisticRegression(max_iter=2000)
            self.model.fit(X_train, y_train)
            
            # Predicciones directas (Clase 0 o 1)
            pred = self.model.predict(X_test)
            
            # Métricas estándar
            acc = accuracy_score(y_test, pred)
            cm = confusion_matrix(y_test, pred)
            cr = classification_report(y_test, pred)
            self.last_cm = cm

            # --- Cálculos Matemáticos Manuales (Logit & Probabilidad) ---
            # Objetivo: Mostrar cómo se llega a la probabilidad desde los coeficientes.
            b0 = float(self.model.intercept_[0])
            b = np.array(self.model.coef_[0])

            # Cálculo del valor Logit (z) sobre el conjunto de test
            # Formula: z = b0 + b1*x1 + b2*x2 ...
            logit_values = np.dot(X_test, b) + b0
            
            # Cálculo de e^z (odds)
            exp_logit = np.exp(logit_values)
            
            # Cálculo de Probabilidad (Sigmoide): P = e^z / (1 + e^z)
            probabilities = exp_logit / (1 + exp_logit)

            # Cálculo de Log-Verosimilitud (Log-Likelihood)
            # Clip evita log(0) que daría error matemático (-inf)
            probs_clipped = np.clip(probabilities, 1e-10, 1 - 1e-10)
            log_likelihood = np.sum(y_test * np.log(probs_clipped) + (1 - y_test) * np.log(1 - probs_clipped))

            # Guardar DataFrame con resultados detallados para exportación
            df_results = pd.DataFrame(X_test, columns=[f"scaled_{c}" for c in X.columns])
            df_results["IngresoReal"] = y_test.values
            df_results["Prediccion"] = pred
            df_results["Probabilidad_Bajo"] = probabilities
            self.last_results_df = df_results

            # --- Reporte en Consola ---
            self.text_results.delete("1.0", tk.END)
            self.text_results.insert(tk.END, "==========================\n")
            self.text_results.insert(tk.END, "=== INFORME DEL MODELO ===\n")
            self.text_results.insert(tk.END, "==========================\n\n")
            self.text_results.insert(tk.END, f"Archivo: {os.path.basename(self.last_file_path) if self.last_file_path else 'N/A'}\n")
            self.text_results.insert(tk.END, f"Variables usadas: {selected_cols}\n\n")
            self.text_results.insert(tk.END, f"Mediana de corte (IngresoTotal): {mediana:.2f}\n")
            self.text_results.insert(tk.END, "Clases: 1 = Ingreso Bajo (< mediana), 0 = Ingreso Alto (>= mediana)\n\n")
            self.text_results.insert(tk.END, f"Exactitud (Accuracy): {acc:.4f}\n\n")
            self.text_results.insert(tk.END, "Matriz de Confusión:\n")
            self.text_results.insert(tk.END, f"{cm}\n\n")
            self.text_results.insert(tk.END, "Reporte de Clasificación:\n")
            self.text_results.insert(tk.END, f"{cr}\n\n")

            # Tabla mejorada de cálculos intermedios (alineada y con log-verosimilitud parcial)
            self.text_results.insert(tk.END, "===============================================================\n")
            self.text_results.insert(tk.END, "=== CÁLCULOS INTERMEDIOS (Muestra de 10 casos del Test Set) ===\n")
            self.text_results.insert(tk.END, "===============================================================\n\n")

            # Encabezados bien alineados
            self.text_results.insert(
                tk.END,
                f"{'Idx':>3} | {'Logit(z)':>10} | {'e^z (Odds)':>12} | {'Prob P(y=1)':>14} | {'LL Parcial':>14}\n"
            )
            self.text_results.insert(
                tk.END,
                f"{'-'*3}-+-{'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*14}\n"
            )

            # Cálculo de log-verosimilitud parcial por caso
            ll_parciales = y_test.values * np.log(probs_clipped) + (1 - y_test.values) * np.log(1 - probs_clipped)

            # Mostrar hasta 10 filas
            for i in range(min(10, len(logit_values))):
                self.text_results.insert(
                    tk.END,
                    f"{i:>3} | {logit_values[i]:>10.4f} | {exp_logit[i]:>12.4f} | {probabilities[i]:>14.4f} | {ll_parciales[i]:>14.4f}\n"
                )

            # Mostrar la suma total al final
            self.text_results.insert(tk.END, f"\nLog-Verosimilitud Total (Log-Likelihood): {log_likelihood:.6f}\n\n")

            # Interpretación simple de coeficientes
            self.text_results.insert(tk.END, "=================================================================\n")
            self.text_results.insert(tk.END, "=== INTERPRETACIÓN DE COEFICIENTES (Variables Estandarizadas) ===\n")
            self.text_results.insert(tk.END, "=================================================================\n\n")
            for cname, coef in zip(X.columns, b):
                efecto = "AUMENTA" if coef > 0 else "DISMINUYE"
                self.text_results.insert(tk.END, f"- {cname}: Coef {coef:.4f}. {efecto} la probabilidad de ser 'Ingreso Bajo'.\n")
            self.text_results.insert(tk.END, "\nNota: Al estar escalados, mayor valor absoluto indica mayor importancia relativa.\n")

            self.status_var.set(f"Modelo entrenado. Accuracy={acc:.4f} - LogLike={log_likelihood:.4f}")
            self.grafico_regresion_logistica() 
            messagebox.showinfo("Entrenamiento finalizado", "Modelo entrenado exitosamente. Revise la consola para detalles.")
        except Exception as e:
            messagebox.showerror("Error durante entrenamiento", str(e))
            self.status_var.set("Error durante entrenamiento")

    #-----------------------------Gráfica de regresión logistica---------------------------
    def grafico_regresion_logistica(self):
        """Muestra un gráfico de regresión logística para una variable seleccionada por el usuario."""
        if self.model is None or self.scaler is None or self.last_results_df is None:
            messagebox.showwarning("Advertencia", "Debe entrenar el modelo antes de graficar.")
            return

        # Pedir variable al usuario
        selected_indices = self.listbox.curselection()
        if len(selected_indices) != 1:
            messagebox.showwarning("Advertencia", "Debe seleccionar UNA variable numérica para graficar su Regresión Logística.")
            return

        var_name = self.listbox.get(selected_indices[0])
        if var_name not in self.encoded_data.columns:
            messagebox.showerror("Error", f"La variable seleccionada no existe: {var_name}")
            return

        if not pd.api.types.is_numeric_dtype(self.encoded_data[var_name]):
            messagebox.showwarning("Advertencia", f"La variable '{var_name}' no es numérica.")
            return

        # Extraer datos
        X_original = self.encoded_data[var_name].copy()
        y = self.encoded_data["IngresoBin"].copy()

        # Reescalar la columna para aplicar el modelo
        single_scaled = self.scaler.transform(self.encoded_data[self.scaler.feature_names_in_])

        # índice de la variable dentro del scaler
        try:
            var_index = list(self.scaler.feature_names_in_).index(var_name)
        except:
            messagebox.showerror("Error", "No se pudo localizar la variable dentro del StandardScaler.")
            return

        # Crear figura y ventana
        win = tk.Toplevel(self.root)
        win.title(f"Regresión Logística - {var_name}")
        win.geometry("1200x600")
        win.state("zoomed")
        win.resizable(False, False)
        win.configure(bg="#f7fbff")

        fig, ax = plt.subplots(figsize=(8,6))

        # -------------------------------
        #     CURVA LOGÍSTICA
        # -------------------------------
        x_vals = np.linspace(X_original.min(), X_original.max(), 300)
        x_vals_scaled = (x_vals - self.scaler.mean_[var_index]) / self.scaler.scale_[var_index]

        # Construir matriz dummy (solo var seleccionada importa)
        X_dummy = np.zeros((300, len(self.scaler.feature_names_in_)))
        X_dummy[:, var_index] = x_vals_scaled

        # Coeficientes
        b0 = float(self.model.intercept_[0])
        b = self.model.coef_[0]

        logits = b0 + np.dot(X_dummy, b)
        probs = 1 / (1 + np.exp(-logits))

        # -------------------------------
        #     PUNTOS REALES
        # -------------------------------
        ax.scatter(
            X_original, y,
            alpha=0.25,
            label="Datos reales (Binarios)",
            color="#0072B2"
        )

        # Curva logística
        ax.plot(
            x_vals,
            probs,
            color="red",
            linewidth=2,
            label="Curva logística estimada"
        )

        # -------------------------------
        #     LÍNEA EN LA MEDIA
        # -------------------------------
        media = X_original.mean()
        ax.axvline(media, color="green", linestyle="--", linewidth=1.5,
                label=f"Media de {var_name}: {media:.2f}")

        # -------------------------------
        #     FORMATO
        # -------------------------------
        ax.set_title(f"Regresión Logística — Variable: {var_name}", fontsize=14)
        ax.set_xlabel(var_name, fontsize=12)
        ax.set_ylabel("Probabilidad de Ingreso Bajo (y=1)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --------------------------- Gráficos y Ventana Visual ---------------------------
    def open_graph_window(self):
        """Abre una ventana secundaria para visualización de datos."""
        if self.encoded_data is None:
            messagebox.showwarning("Advertencia", "Cargue un archivo primero.")
            return
        win = tk.Toplevel(self.root)
        win.title("Dashboard de Visualización")
        win.geometry("1200x600")
        win.state("zoomed")
        win.resizable(False, False)
        win.configure(bg="#f7fbff")

        # Panel de botones gráficos
        ctrl_frame = tk.Frame(win, bg="#f7fbff")
        ctrl_frame.pack(fill="x", padx=8, pady=6)

        tk.Button(ctrl_frame, text="Histograma IngresoTotal", command=self.plot_hist_ingreso).pack(side="left", padx=6)
        tk.Button(ctrl_frame, text="Gráficos de Dispersión (IngresoTotal)", command=self.grafico_dispersion).pack(side="left", padx=6)
        tk.Button(ctrl_frame, text="Barras - Promedio por Zona", command=self.plot_bar_promedio_zona).pack(side="left", padx=6)
        tk.Button(ctrl_frame, text="Pie - Distribución DiasSemana", command=self.plot_pie_dias).pack(side="left", padx=6)
        tk.Button(ctrl_frame, text="Gráficos de Dispersión (Elegir)", command=self.plot_scatter_select).pack(side="left", padx=6)
        tk.Button(ctrl_frame, text="Matriz de Confusión", command=lambda: self.plot_confusion(self.last_cm)).pack(side="left", padx=6)

        # Área de dibujo (Matplotlib backend)
        fig_frame = tk.Frame(win, bg="#f7fbff")
        fig_frame.pack(fill="both", expand=True, padx=8, pady=6)

        self.fig, self.ax = plt.subplots(figsize=(8,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear_axes(self):
        """Limpia correctamente todo el gráfico."""
        self.fig.clf()            # Limpia toda la figura
        self.ax = self.fig.add_subplot(111)  # Crea un nuevo eje limpio

    def plot_hist_ingreso(self):
        if self.encoded_data is None:
            return
        self.clear_axes()
        color = getattr(self, "plot_color", "#0b5394")
        self.ax.hist(self.encoded_data["IngresoTotal"].dropna(), bins=25, color=color, edgecolor="white")
        self.ax.set_title("Histograma de IngresoTotal")
        self.ax.set_xlabel("IngresoTotal")
        self.ax.set_ylabel("Frecuencia")
        self.canvas.draw()
    
    def grafico_dispersion(self):
        """Gráfico de dispersión: si hay >1 variable, abre ventana con botones a la derecha para ver un gráfico a la vez."""
        if self.encoded_data is None:
            messagebox.showwarning("Advertencia", "Cargue un archivo primero.")
            return

        selected_indices = self.listbox.curselection()
        if len(selected_indices) < 1:
            messagebox.showwarning("Aviso", "Seleccione al menos 1 columna para graficar.")
            return

        import seaborn as sns
        import math

        selected_cols = [self.listbox.get(i) for i in selected_indices]
        y_col = "IngresoTotal"
        n = len(selected_cols)

        # Caso n == 1: reproducir comportamiento anterior (ventana única con scatter)
        if n == 1:
            self.clear_axes()
            ax = self.ax
            sns.scatterplot(
                data=self.encoded_data,
                x=selected_cols[0],
                y=y_col,
                hue="IngresoBinario",
                palette=[self.plot_color, '#000000'] if "IngresoBinario" in self.encoded_data.columns else [self.plot_color],
                ax=ax
            )
            ax.set_title(f"{selected_cols[0]} vs {y_col} (coloreado por clase)")
            self.canvas.draw()
            return

        # Si n > 1: crear ventana nueva con canvas + botones a la derecha
        win = tk.Toplevel(self.root)
        win.title("Gráficos de Dispersión — Vista por Variable")
        win.geometry("1200x600")
        win.state("zoomed")
        win.resizable(False, False)
        win.configure(bg="#f7fbff")

        # frames
        left_frame = tk.Frame(win, bg="#f7fbff")
        left_frame.pack(side="left", fill="both", expand=True)
        right_frame = tk.Frame(win, bg="#f7fbff", width=220)
        right_frame.pack(side="right", fill="y")

        # figura matplolib en left_frame
        fig, ax = plt.subplots(figsize=(8,6))
        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # función para pintar una variable concreta
        def plot_single(x_col):
            ax.clear()
            # si hay IngresoBinario, colorear por clase usando tonos derivados
            if "IngresoBinario" in self.encoded_data.columns:
                pal = [self.plot_color, '#000000']
                sns.scatterplot(data=self.encoded_data, x=x_col, y=y_col, hue="IngresoBinario", palette=pal, ax=ax, alpha=0.7)
                ax.legend(title="Clase")
            else:
                ax.scatter(self.encoded_data[x_col], self.encoded_data[y_col], color=self.plot_color, alpha=0.7)
            ax.set_title(f"{x_col} vs {y_col}")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            canvas.draw()

        # crear botones en right_frame
        lbl = tk.Label(right_frame, text="Variables (mostrar 1):", bg="#f7fbff")
        lbl.pack(pady=(8,4))
        for col in selected_cols:
            b = tk.Button(right_frame, text=col, width=25, command=lambda c=col: plot_single(c))
            b.pack(pady=3)

        # botón para cerrar/volver (opcional)
        tk.Button(right_frame, text="Cerrar", command=win.destroy).pack(side="bottom", pady=8)

        # plot primer gráfico por defecto
        plot_single(selected_cols[0])

    def plot_bar_promedio_zona(self):
        if self.encoded_data is None: return
        if "Zona" in self.data.columns:
            df = self.data.groupby("Zona")["IngresoTotal"].mean().reset_index()
            x = df["Zona"]
            y = df["IngresoTotal"]
            self.clear_axes()
            color = getattr(self, "plot_color", "#0b5394")
            bars = self.ax.bar(x, y, color=color, edgecolor="white")
            self.ax.set_title("Ingreso Promedio por Zona")
            self.ax.set_ylabel("Ingreso Promedio")
            self.ax.set_xticklabels(x, rotation=45, ha="right")
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            messagebox.showwarning("Aviso", "La columna 'Zona' original no está disponible.")

    def plot_pie_dias(self):
        if self.encoded_data is None: return
        if "DiaSemana" in self.data.columns:
            counts = self.data["DiaSemana"].value_counts()
            self.clear_axes()
            # obtener tonos (si no existen, generarlos)
            shades = getattr(self, "_pie_shades", None)
            if shades is None or len(shades) < len(counts):
                shades = self.generate_shades(self.plot_color, max(6, len(counts)))
            # limitar a n
            colors = shades[:len(counts)]
            wedges, texts, autotexts = self.ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
            # mejorar contraste del texto del % si es necesario
            for txt in autotexts:
                txt.set_color("black")
            self.ax.set_title("Distribución por Día de la Semana")
            self.ax.axis("equal")
            self.fig.tight_layout()
            self.canvas.draw()
        else:
            messagebox.showwarning("Aviso", "La columna 'DiaSemana' original no está disponible.")

    def plot_scatter_select(self):
        if self.encoded_data is None: return
        numeric_cols = [c for c in self.encoded_data.columns if pd.api.types.is_numeric_dtype(self.encoded_data[c]) and c!="IngresoTotal"]
        if len(numeric_cols) < 2:
            messagebox.showwarning("Aviso", "No hay suficientes columnas numéricas para scatter.")
            return

        sel = simpledialog.askstring("Scatter", f"Ingrese 2 columnas separadas por coma (ej: {numeric_cols[0]},{numeric_cols[1]})")
        if not sel: return

        parts = [s.strip() for s in sel.split(",")]
        if len(parts) != 2 or parts[0] not in numeric_cols or parts[1] not in numeric_cols:
            messagebox.showerror("Error", "Columnas inválidas o no encontradas.")
            return

        xcol, ycol = parts[0], parts[1]
        self.clear_axes()
        color = getattr(self, "plot_color", "#0b5394")
        self.ax.scatter(self.encoded_data[xcol], self.encoded_data[ycol], color=color, alpha=0.6)
        self.ax.set_xlabel(xcol); self.ax.set_ylabel(ycol)
        self.ax.set_title(f"Dispersión: {xcol} vs {ycol}")
        self.canvas.draw()

    def plot_confusion(self, cm):
        """Dibuja la matriz de confusión usando el color elegido por el usuario."""
        if cm is None:
            messagebox.showwarning("Aviso", "No hay matriz de confusión. Entrene el modelo primero.")
            return

        self.clear_axes()
        cm = np.array(cm)

        # Usar el colormap derivado del color elegido
        cmap = getattr(self, "_heatmap_cmap", None)
        if cmap is None:
            cmap = self.generate_diverging_cmap(self.plot_color)

        # Dibujar matriz
        im = self.ax.imshow(cm, cmap=cmap)

        # Anotaciones sobre cada celda
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                text_color = "black"
                self.ax.text(j, i, value, ha="center", va="center", color=text_color, fontsize=12)

        self.ax.set_title("Matriz de Confusión", fontsize=14)
        self.ax.set_xlabel("Predicción")
        self.ax.set_ylabel("Real")

        # Barra de color
        self.fig.colorbar(im, ax=self.ax)

        self.fig.tight_layout()
        self.canvas.draw()

    # --------------------------- Popup Matriz Confusión ---------------------------
    def show_confusion_matrix(self, cm, mini=False):
        """Muestra una ventana emergente pequeña con la matriz de confusión tras entrenar."""
        win = tk.Toplevel(self.root)
        win.title("Matriz de Confusión")
        win.geometry("480x420" if not mini else "360x320")
        fig, ax = plt.subplots(figsize=(4.5,4) if not mini else (3.5,3))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Matriz de Confusión")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack()

    # --------------------------- Exportación ---------------------------
    def save_results_csv(self):
        """Exporta el DataFrame de resultados (predicciones y probabilidades) a CSV."""
        if self.last_results_df is None:
            messagebox.showwarning("Advertencia", "No hay resultados para guardar. Entrene el modelo primero.")
            return
        save = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if save:
            try:
                self.last_results_df.to_csv(save, index=False)
                messagebox.showinfo("Guardado", f"Resultados guardados en: {save}")
            except Exception as e:
                messagebox.showerror("Error al guardar", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictApp(root)
    root.mainloop()
