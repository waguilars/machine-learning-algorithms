""" Red neuronal """

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

print('-----------------------  IRIS --------------------\n')

x, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    random_state=0, max_iter=1000).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy: ', clf.score(X_test, y_test))

precision = precision_score(y_test, y_pred, average=None)
print('presicion: ', precision)

recall = recall_score(y_test, y_pred, average=None)
print('recall: ', recall)

f1 = f1_score(y_test, y_pred, average=None)
print('F1 Measure: ', f1)
###################################################################
print('-----------------------  WINE --------------------\n')

x, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    max_iter=1000, random_state=0).fit(X_train, y_train)


y_pred = clf.predict(X_test)
print('Accuracy: ', clf.score(X_test, y_test))

precision = precision_score(y_test, y_pred, average=None)
print('presicion: ', precision)

recall = recall_score(y_test, y_pred, average=None)
print('recall: ', recall)

f1 = f1_score(y_test, y_pred, average=None)
print('F1 Measure: ', f1)
