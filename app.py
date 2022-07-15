# Deploy pHs Machine Learning Model With Flask  
# Author: Mahmoud Basseem I. Mohamed
# affiliation: Chemistry Department, Faculty of Science, Al-Azhar University, Nasr City, Cairo, P.O.11884, Egypt
# Date: 15-07-2022
# Version: 0.1 
#-----------------------------------------------------------------------------------------------------------------------
# Importing the libraries
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, abort, send_from_directory
import numpy as np  # Importing the numpy library
import pandas as pd  # Importing the pandas library
import pickle
import sklearn  # Importing the sklearn library
import cv2  # Importing the cv2 library

# creating the flask app
app = Flask(__name__)

# loading the model
model = pickle.load(open('model/KNeighborsRegressor.pkl', 'rb'))

# creating the route for the home page
@app.route('/')
def index():
    return render_template('index.html')
# creating the route for the predict page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # getting the image from the request
        image = request.files['image']
        # save the images to the local directory
        image.save('./static/images/image.png')
        # loading the image
        img = cv2.imread('./static/images/image.png')
        # extracting the features from the image RGB values at 25,25 axis
        b = img[25, 25, 0]
        g = img[25, 25, 1]
        r = img[25, 25, 2]
        # creating the feature vector
        X = np.array([r, g, b])
        # reshaping the feature vector
        X = X.reshape(-1, 3)
        # predicting the image
        prediction = model.predict(X)
        # returning the prediction
        return render_template('predict.html', prediction_text='The pH value is: {}'.format(prediction))
        
# running the app
if __name__ == '__main__':
    app.run(debug=True)
#-----------------------------------------------------------------------------------------------------------------------


