import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Función para cargar los archivos manualmente
def cargar_archivos():
    global customers_df, packages_df, shipping_companies_df, preferences_df
    
    # Seleccionar los archivos mediante el cuadro de diálogo
    customer_file = filedialog.askopenfilename(title="Seleccionar archivo de clientes", filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
    package_file = filedialog.askopenfilename(title="Seleccionar archivo de paquetes", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    shipping_file = filedialog.askopenfilename(title="Seleccionar archivo de compañías de envío", filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
    preferences_file = filedialog.askopenfilename(title="Seleccionar archivo de preferencias", filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*")))
    
    # Cargar los archivos en los DataFrames correspondientes
    customers_df = pd.read_excel(customer_file)
    packages_df = pd.read_csv(package_file)
    shipping_companies_df = pd.read_excel(shipping_file)
    preferences_df = pd.read_excel(preferences_file)

# Función para entrenar el modelo de red neuronal
def entrenar_modelo():
    global nn_clf
    
    # Merge packages_df with shipping_companies_df based on 'CompanyID'
    merged_df = pd.merge(packages_df, shipping_companies_df, on='CompanyID', how='left')
    
    # Assume the target variable is the preferred shipping company for each customer
    # Predict based on the best cost, speed, and delivery status
    preferred_company_indices = merged_df.groupby('PackageID')['ShippingCost'].idxmin()
    merged_df['PreferredCompanyID'] = merged_df.loc[preferred_company_indices, 'CompanyID'].values
    merged_df['PreferredCompanyID'] = np.where(merged_df['ShippingSpeed'] == 'Express', merged_df['PreferredCompanyID'], merged_df['CompanyID'])
    
    # Features for the neural network
    features_nn = ['ShippingCost', 'ShippingSpeed', 'Weight']
    
    # Drop rows with NaN values in the features and target variable
    merged_df = merged_df.dropna(subset=features_nn + ['PreferredCompanyID'])
    
    # Encode categorical feature 'ShippingSpeed'
    merged_df['ShippingSpeed'] = merged_df['ShippingSpeed'].map({'Standard': 0, 'Express': 1})
    
    # Split the data into training and testing sets
    X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
        merged_df[features_nn], merged_df['PreferredCompanyID'], test_size=0.2, random_state=42)
    
    # Train a neural network classifier
    scaler = StandardScaler()
    X_train_nn_scaled = scaler.fit_transform(X_train_nn)
    nn_clf = MLPClassifier(hidden_layer_sizes=(6, 6, 6), max_iter=1000, random_state=42)
    nn_clf.fit(X_train_nn_scaled, y_train_nn)
    
    messagebox.showinfo("Entrenamiento Completo", "El modelo ha sido entrenado con éxito.")
    
    # Guardar el modelo entrenado
    joblib.dump(nn_clf, 'modelo_entrenado.pkl')
    messagebox.showinfo("Modelo Guardado", "El modelo entrenado ha sido guardado como 'modelo_entrenado.pkl'.")

# Función para cargar un modelo previamente entrenado
def cargar_modelo():
    global nn_clf
    
    try:
        nn_clf = joblib.load('modelo_entrenado.pkl')
        messagebox.showinfo("Modelo Cargado", "El modelo entrenado ha sido cargado correctamente.")
    except FileNotFoundError:
        messagebox.showerror("Archivo No Encontrado", "No se encontró ningún modelo entrenado. Por favor, entrena un modelo primero.")

# Función para detener el entrenamiento del modelo
def detener_entrenamiento():
    global nn_clf
    
    if nn_clf is not None:
        nn_clf = None
        messagebox.showinfo("Entrenamiento Detenido", "El entrenamiento del modelo ha sido detenido.")
    else:
        messagebox.showinfo("Modelo No Entrenado", "No hay ningún modelo en proceso de entrenamiento.")

# Función para ingresar datos
def ingresar_datos():
    if nn_clf is None:
        messagebox.showinfo("Modelo No Entrenado", "Por favor, primero entrena el modelo antes de ingresar los datos.")
        return
    
    # Abrir ventana para ingresar datos
    ventana_ingreso_datos = tk.Toplevel(root)
    ventana_ingreso_datos.title("Ingresar Datos de Envío")
    
    tk.Label(ventana_ingreso_datos, text="Dimensiones del Paquete (dcm^3):").grid(row=0, column=0, padx=10, pady=5)
    entry_dimensiones = tk.Entry(ventana_ingreso_datos)
    entry_dimensiones.grid(row=0, column=1, padx=10, pady=5)
    
    tk.Label(ventana_ingreso_datos, text="Velocidad de Envío (Standard/Express):").grid(row=1, column=0, padx=10, pady=5)
    entry_velocidad_envio = tk.Entry(ventana_ingreso_datos)
    entry_velocidad_envio.grid(row=1, column=1, padx=10, pady=5)
    
    tk.Label(ventana_ingreso_datos, text="Peso del Paquete (kg):").grid(row=2, column=0, padx=10, pady=5)
    entry_peso_paquete = tk.Entry(ventana_ingreso_datos)
    entry_peso_paquete.grid(row=2, column=1, padx=10, pady=5)
    
    tk.Button(ventana_ingreso_datos, text="Calcular Mejor Compañía", command=lambda: calcular_mejor_compania(entry_dimensiones.get(), entry_velocidad_envio.get(), entry_peso_paquete.get())).grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Función para calcular la mejor compañía de envío
def calcular_mejor_compania(dimensiones, velocidad_envio, peso_paquete):
    if nn_clf is None:
        messagebox.showinfo("Modelo No Entrenado", "Por favor, primero entrena el modelo antes de ingresar los datos.")
        return
    
    try:
        dimensiones = float(dimensiones)
        if velocidad_envio.lower() not in ['standard', 'express']:
            raise ValueError("La velocidad de envío debe ser 'Standard' o 'Express'.")
        peso_paquete = float(peso_paquete)
    except ValueError as e:
        messagebox.showerror("Error en Entrada", str(e))
        return
    
    velocidad_codificada = 1 if velocidad_envio.lower() == 'express' else 0
    
    # Realizar la predicción con el modelo de red neuronal
    predicciones = nn_clf.predict([[dimensiones, velocidad_codificada, peso_paquete]])
    compania_recomendada = shipping_companies_df.loc[shipping_companies_df['CompanyID'] == predicciones[0], 'CompanyName'].values[0]
    
    # Mostrar la compañía recomendada al usuario
    messagebox.showinfo("Recomendación de Envío", f"Se recomienda enviar su paquete con {compania_recomendada}.")

# Crear la ventana principal de la aplicación
root = tk.Tk()
root.title("Recomendación de Envío")

# Botón para cargar los archivos manualmente
btn_cargar_archivos = tk.Button(root, text="Cargar Archivos", command=cargar_archivos)
btn_cargar_archivos.pack(pady=10)

# Botón para cargar un modelo previamente entrenado
btn_cargar_modelo = tk.Button(root, text="Cargar Modelo", command=cargar_modelo)
btn_cargar_modelo.pack(pady=5)

# Botón para empezar el entrenamiento del modelo
btn_entrenar_modelo = tk.Button(root, text="Empezar Entrenamiento", command=entrenar_modelo)
btn_entrenar_modelo.pack(pady=5)

# Botón para detener el entrenamiento del modelo
btn_detener_entrenamiento = tk.Button(root, text="Detener Entrenamiento", command=detener_entrenamiento)
btn_detener_entrenamiento.pack(pady=5)

# Botón para ingresar datos
btn_ingresar_datos = tk.Button(root, text="Ingresar Datos", command=ingresar_datos)
btn_ingresar_datos.pack(pady=5)

# Inicializar el modelo como nulo
nn_clf = None

# Iniciar el bucle principal
root.mainloop()
