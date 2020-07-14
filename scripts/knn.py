""" KNN """
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


x, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
predicted = knn.predict(X_test)
acc = knn.score(X_test, y_test)
print(acc)
print(predicted)
