# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:59:56 2020

@author: Sunshine
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
model = pickle.load(open('internshala.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction >= 90.0:
        output = "Alpha"
    elif prediction >= 80.0  and prediction < 90.0:
        output = "Beta"
    elif prediction >= 70.0 and prediction < 80.0:
        output = "Gama"
    else :
        output = "No Campus Recruitment"
    return render_template('index.html', prediction_text='Campus Recruitment : {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)