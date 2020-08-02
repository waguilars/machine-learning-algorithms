###### wine logist regresion 

import numpy as np
import math as ma
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
# Importamos el conjunto de datos
def regresion(wine):
    dataset =wine
    X = dataset.iloc[:, 1:13].values
    y = dataset.iloc[:, 0].values
# Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
# Applying PCA
    pca = PCA(n_components = 2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
# Explicación de la varianza
# Creamos un vector con el porcentaje de influencia de la varianza 
# para las dos variables resultantes del conjunto de datos
    explained_variance = pca.explained_variance_ratio_
# Fitting Logistic Regression to the Training set
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)

# Predicting the Test set results
    y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    # plt.show()
    # print(y_pred)
    # print('Son {} datos para entrenamiento y {} datos para prueba'.format(X_train.shape[0], X_test.shape[0]))
    algoritmo = LogisticRegression()
    algoritmo.fit(X_train, y_train)
    y_predic = algoritmo.predict(X_test)
    # print('Precisión Regresión Logística: {}'.format(algoritmo.score(X_train, y_train)))

    mae = mean_absolute_error(y_test, y_predic)
# print("=================================")
# print("MAE:", mae)
# print("=================================\n")
    mse = mean_squared_error(y_test, y_predic)
# print("=================================")
# print("MSE:", mse)
# print("=================================\n")


    rmse = ma.sqrt(mse)
# print("=================================")
# print("RMSE:", rmse)
# print("=================================\n")

    r2 = algoritmo.score(X_test, y_test)
# print("=================================")
# print("R2:", r2)
# print("=================================\n")
    r2a = 1-(1-r2)*(len(y_predic)-1)/(len(y_predic)-len(algoritmo.coef_)-1)
    results = [mae,mse,rmse,r2,r2a]
    return results