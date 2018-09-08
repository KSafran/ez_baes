from flask import Flask
from . import iris
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Welcome to EZ Baes'

@app.route('/train_iris')
def train_iris():
    iris.fit_iris()
    return 'Training Model...'
