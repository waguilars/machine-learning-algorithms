from flask import Flask,Response, jsonify, render_template
from sklearn.datasets import load_iris, load_wine
from scripts import knn,random_forest



knn_iris = knn.knn_iris(load_iris)
knn_wine = knn.knn_wine(load_wine)
Rf_iris = random_forest.forest_iris(load_iris())


app = Flask(__name__)

@app.route('/iris',methods = ['Get'])
def iris():
    return render_template('iris.html',active=['','active',''],knn=knn_iris)

@app.route('/wine',methods = ['Get'])
def wine():
    return render_template('wine.html',active=['','','active'],knn=knn_wine)

@app.route('/')
@app.route('/index',methods = ['Get'])
def index():
    return render_template('index.html',active=['active','',''])




if __name__ == "__main__":
    app.run(debug=True, port=4000)


