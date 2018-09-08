from flask import Flask
from . import iris
app = Flask(__name__)

@app.route('/')
def hello_world():
    iris.fit_iris()
    return 'Training Model...'
