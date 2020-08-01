""" KNN """
from sklearn.datasets import load_iris, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score


x, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
predicted = knn.predict(X_test)
acc = knn.score(X_test, y_test)

print('Accuracy:', acc)

presicion = precision_score(y_test, predicted, average=None)
print('Presicion:', presicion)

recall = recall_score(y_test, predicted, average=None)
print('Recall:', recall)

f1 = f1_score(y_test, predicted, average=None)
print('F1 measure:', f1)


print('-----------------------  WINE --------------------\n')
x, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
predicted = knn.predict(X_test)
acc = knn.score(X_test, y_test)

print('Accuracy:', acc)

presicion = precision_score(y_test, predicted, average=None)
print('Presicion:', presicion)

recall = recall_score(y_test, predicted, average=None)
print('Recall:', recall)

f1 = f1_score(y_test, predicted, average=None)
print('F1 measure:', f1)
