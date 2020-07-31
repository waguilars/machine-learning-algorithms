""" Red neuronal """

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
x, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)


clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 3),
                    random_state=1, max_iter=300).fit(X_train, y_train)

print(clf.predict(X_test))
print(clf.score(X_test, y_test))
