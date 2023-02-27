# flask app to to deploy the model
# import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# create flask app
app = Flask(__name__)
# load the model
model = pickle.load(open('model.pkl', 'rb'))

# create a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# create a route for the prediction page
@app.route('/predict_api',methods=['POST'])
def predict_api():
    # get the values from the form
    data = request.get_json(force=True)
    # convert the values to a numpy array