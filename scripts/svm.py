""" Maquinas de soporte vectorial """

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt



def get_recall(data):
    test = data.sum(axis=1)
    diagonal = np.diag(data)
    res = np.divide(diagonal, test)
    return res


def get_precision(data):
    test = data.sum(axis=0)
    diagonal = np.diag(data)
    res = np.divide(diagonal, test)
    return res


def get_fmeasure(conf_mtx):
    presicion = get_precision(conf_mtx)
    recall = get_recall(conf_mtx)
    num = presicion*recall
    den = presicion+recall
    f1 = 2*(num/den)
    return f1

def get_accuracy(conf_mtx):
    tp_tn = sum(np.diag(conf_mtx))
    total = np.sum(conf_mtx)
    return tp_tn / total



X, y = load_iris(return_X_y=True)

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=0)

clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

matrix_confusion = confusion_matrix(y_test,y_pred)

# ==========================
#           PRESICION
# ==========================
presicion = get_precision(matrix_confusion)
print("presicion: \n\t", presicion)

# ==========================
#           RECALL
# ==========================
recall = get_recall(matrix_confusion)
print("recall: \n\t", recall)

# ==========================
#         ACCURACY
# ==========================
accuracy = get_accuracy(matrix_confusion)
print("accuracy: \n\t", accuracy)

# ==========================
#         F-MEASURE
# ==========================
f1 = get_fmeasure(matrix_confusion)
print("f-measure: \n\t", f1)