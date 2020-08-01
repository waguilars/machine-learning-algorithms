""" Regresion logistica """
import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
#Análizamos los datos que tenemos disponibles
print('Información del dataset:')
iris = pd.read_csv("./data/Iris.csv")
#Eliminamos la primera columna ID
iris = iris.drop('Id',axis=1)
print(iris.head())
#Grafico Sepal - Longitud vs Ancho
fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='blue', label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='green', label='Versicolor', ax=fig)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='SepalLengthCm', y='SepalWidthCm', color='red', label='Virginica', ax=fig)
fig.set_xlabel('Sépalo - Longitud')
fig.set_ylabel('Sépalo - Ancho')
fig.set_title('Sépalo - Longitud vs Ancho')
plt.show()

#Grafico Pétalo - Longitud vs Ancho
fig = iris[iris.Species == 'Iris-setosa'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='Versicolor', ax=fig)
iris[iris.Species == 'Iris-virginica'].plot(kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='red', label='Virginica', ax=fig)
fig.set_xlabel('Pétalo - Longitud')
fig.set_ylabel('Pétalo - Ancho')
fig.set_title('Pétalo Longitud vs Ancho')
plt.show()

#Separar todos los datos con las características y las etiquetas o resultados
X = np.array(iris.drop(['Species'], 1))
y = np.array(iris['Species'])

#Separar los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))

#Modelo de Regresión Logística
algoritmo = LogisticRegression()
algoritmo.fit(X_train, y_train)
y_predic = algoritmo.predict(X_test)
print('Precisión Regresión Logística: {}'.format(algoritmo.score(X_train, y_train)))

print(y_predic)
y_predic=[1,2,2,2,1,0,1,2,0,0,2,1,0,0,1,2,1,1,1,0,1,2,1,2,1,2,0,2,0,0]
print(y_predic)
y_test=[1,2,0,0,1,1,1,0,2,1,2,2,2,2,2,0,2,1,0,1,1,0,2,0,2,1,2,0,0,1]
print(y_test)
mae = mean_absolute_error(y_test, y_predic)
print("=================================")
print("MAE:", mae)
print("=================================\n")

mse = mean_squared_error(y_test, y_predic)
print("=================================")
print("MSE:", mse)
print("=================================\n")
rmse = ma.sqrt(mse)
print("=================================")
print("RMSE:", rmse)
print("=================================\n")

r2 = algoritmo.score(X_test, y_test)
print("=================================")
print("R2:", r2)
print("=================================\n")

r2a = 1-(1-r2)*(len(y_predic)-1)/(len(y_predic)-len(algoritmo.coef_)-1)
print("=================================")
print("R2 Ajustado:", r2a)
print("=================================\n")



