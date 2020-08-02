from flask import Flask,Response, jsonify, render_template
from sklearn.datasets import load_iris, load_wine
from scripts import knn,random_forest,NB_and_RF,neural_network,naive_bayes,svm
from scripts.logistic_regresion import logistic_regression,logistic_regression_wine
import pandas as pd
wine=pd.read_csv('./scripts/data/Wine.csv',header=None)
iris = pd.read_csv("./scripts/data/iris.csv")
wine1=pd.read_csv('./scripts/data/Wine.csv')

knn_iris = knn.knn_iris(load_iris)
knn_wine = knn.knn_wine(load_wine)

Rf_iris = random_forest.forest_iris(load_iris())
bayes_iris=naive_bayes.baiyes_iris(load_iris)
Rf_NB_wine= NB_and_RF.forest_bayes(wine)

logic_iris=logistic_regression.regresion(iris)
logic_wine=logistic_regression_wine.regresion(wine1)

neuronal_iris = neural_network.Neuronal_iris(load_iris)
neuronal_wine = neural_network.Neuronal_Wine(load_wine)

svm__iris=svm.svm_iris(load_iris)
svm_wine=svm.svm_wine(load_wine)

app = Flask(__name__)

@app.route('/iris',methods = ['Get'])
def iris():
    return render_template('iris.html',active=['','active',''],knn=knn_iris,rf=Rf_iris,nb=bayes_iris,nn=neuronal_iris,lr=logic_iris,svm=svm__iris)

@app.route('/wine',methods = ['Get'])
def wine():
    return render_template('wine.html',active=['','','active'],knn=knn_wine,rf=Rf_NB_wine[1],nb=Rf_NB_wine[0],nn=neuronal_wine,lr=logic_wine,svm=svm_wine)

@app.route('/')
@app.route('/index',methods = ['Get'])
def index():
    return render_template('index.html',active=['active','',''])




if __name__ == "__main__":
    app.run(debug=True, port=4000)


