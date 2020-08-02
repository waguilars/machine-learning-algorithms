import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

def baiyes_iris(dataset):
    iris = dataset()
    iris.target_names
    iris.feature_names
    iris.data[0:150]
    iris.target

    data=pd.DataFrame({
        'sepal length':iris.data[:,0],
        'sepal width':iris.data[:,1],
        'petal length':iris.data[:,2],
        'petal width':iris.data[:,3],
        'species':iris.target
    })
    data.head()

    X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # caracteristicas
    y=data['species']  # etiquetas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 82)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    nvclassifier = GaussianNB()
    nvclassifier.fit(X_train, y_train)

    y_pred = nvclassifier.predict(X_test)
    # print(y_pred)

    y_compare = np.vstack((y_test,y_pred)).T

    y_compare[:5,:]

    cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    a = cm.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred +=cm[row,c]
            else:
                falsePred += cm[row,c]
    return corrPred/(cm.sum())