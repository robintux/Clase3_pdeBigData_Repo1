# %% Cargamos modulos y datos 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
# import sympy

# Datos 
from sklearn.datasets import load_digits
digits = load_digits()

# Modulos para ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Agregamos un conjuntoo de datos a analizar para realizar una regresion
# logistica multiclase

# %% Veamos algo de los datos 

# Lista de metodos
dir(digits)

# Descripcion del dataset
print(digits["DESCR"])


# data 
digits["data"]
digits["data"].shape
digits["data"][0]
digits["data"][0].shape

# images
digits["images"]
digits["images"].shape
digits["images"][0]


digits["target"]
digits["target"][0]
digits["target_names"][0]

# visualicemos la primera observacion
plt.imshow(digits["images"][999], cmap = plt.cm.gray)
digits["target"][999]


# %% Regresion Logistica Multiclase

# Particionamiento de los datos (digits) en train / test
x_train, x_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target, 
                                                    test_size = 0.25)

# COnstruccion del modelo  : Instanciamos la clase LogisticRegression
# Luego de una primera ejecucion, nos dimos cuenta de que hay un warning 
# STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
model1_digits = LogisticRegression(max_iter = 3000)

# Ajustamos el modelo a nuestro conjunto de datos de entrenamiento 
model1_digits.fit(x_train, y_train)

# Hagamos predicciones con el dataset de testeo
model1_digits.predict(x_test)


# %% Hagamos predicciones 

# El metodo score 
score = model1_digits.score(x_test, y_test)
print(score)

# Calculemos pronosticos con el modelo 
Forecast1 = model1_digits.predict(x_test)

# Matriz de confusion
from sklearn import metrics
metrics.confusion_matrix(y_test, Forecast1)




































