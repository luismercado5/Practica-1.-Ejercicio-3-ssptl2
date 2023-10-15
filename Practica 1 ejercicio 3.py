# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 21:20:18 2023

@author: luis mercado
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Lectura del archivo de datos
datos = pd.read_csv('concentlite.csv', header=None)

# Extracción de las entradas y las salidas de los datos
entradas = datos.iloc[:, :-1].values
salidas = datos.iloc[:, -1].values

# Definición de parámetros
num_capas = 5  # Número de capas ocultas (ajusta según tus necesidades)
neuronas_por_capa = [10, 5, 3]  # Número de neuronas para cada capa oculta (ajusta según tus necesidades)
tasa_aprendizaje = 0.01  # Tasa de aprendizaje (ajusta según tus necesidades)
max_iter = 5000  # Número máximo de iteraciones (ajusta según tus necesidades)

# Creación del perceptrón multicapa
mlp = MLPClassifier(hidden_layer_sizes=tuple(neuronas_por_capa), learning_rate_init=tasa_aprendizaje, max_iter=max_iter)

# División del conjunto de datos en entrenamiento y prueba
entradas_entrenamiento, entradas_prueba, salidas_entrenamiento, salidas_prueba = train_test_split(
    entradas, salidas, test_size=0.1, random_state=42
)

# Entrenamiento del perceptrón multicapa
mlp.fit(entradas_entrenamiento, salidas_entrenamiento)

# Evaluación del perceptrón multicapa en los datos de prueba
porcentaje_acierto = mlp.score(entradas_prueba, salidas_prueba)
print("Porcentaje de acierto en los datos de prueba:", porcentaje_acierto * 100, "%")

# Clasificación de todos los puntos del plano para visualizar la superficie de decisión
x_min, x_max = entradas[:, 0].min() - 1, entradas[:, 0].max() + 1
y_min, y_max = entradas[:, 1].min() - 1, entradas[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Gráfico de la superficie de decisión y los puntos de datos
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.scatter(entradas[:, 0], entradas[:, 1], c=salidas, edgecolors='k', cmap=plt.cm.bwr)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Perceptrón Multicapa ')
plt.show()
