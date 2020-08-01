from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def forest_iris(dataset):
    iris = dataset
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 30% test

    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)

    y_pred=clf.predict(X_test)

    return metrics.accuracy_score(y_test, y_pred)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# species_idx = clf.predict([[3, 5, 4, 2]])[0]

# print(iris.target_names[species_idx])


# feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
# feature_imp

# sns.barplot(x=feature_imp, y=feature_imp.index)

# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()
