# flask app to to deploy the model
# import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# create flask app
app = Flask(__name__)

# load the model and scaler
model = pickle.load(open('model.pkl', 'rb')) 
scalar = pickle.load(open('scaler.pkl', 'rb'))


# create a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# create a route for the prediction page
@app.route('/predict_api',methods=['POST'])
def predict_api():
    # get the values from the form
    data = request.get_json(force=True)
    print(data)
    # convert the values to a numpy array
    reshaped_data = np.array(list(data['data'].values()))
    print(reshaped_data)
    
    # scale the data
    transformed_data=scalar.transform(reshaped_data.reshape(1,-1))
    
    # make the prediction
    prediction = model.predict(transformed_data)
    print(prediction[0])

    # return the prediction
    output = prediction[0]
    return jsonify(output)

# run the app
if __name__ == "__main__":
    app.run(debug=True)

# Path: home.html
