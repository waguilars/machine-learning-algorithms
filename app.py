from flask import Flask,Response, jsonify, render_template


app = Flask(__name__)

@app.route('/iris',methods = ['Get'])
def iris():
    return render_template('iris.html',active=['','active',''])

@app.route('/wine',methods = ['Get'])
def wine():
    return render_template('wine.html',active=['','','active'])

@app.route('/')
@app.route('/index',methods = ['Get'])
def index():
    return render_template('index.html',active=['active','',''])




if __name__ == "__main__":
    app.run(debug=True, port=4000)


