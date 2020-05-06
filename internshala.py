# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 19:55:57 2020

@author: Sunshine
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

dataset = pd.read_csv(r"student.csv")

df = pd.DataFrame(dataset)

df['Percentage'] = (( df['Maths']+df['Physics']+df['Chemistry'] ) / 300) * 100

dataset = df

X = dataset.iloc[:, [0,1,2]].values

y = dataset.iloc[:, [4]].values

from sklearn.tree import DecisionTreeRegressor

classifier = DecisionTreeRegressor()
classifier.fit(X, y)

import pickle

pickle.dump(classifier, open('internshala.pkl', 'wb'))

model = pickle.load(open('internshala.pkl', 'rb'))

print(model.predict([[90, 89, 89]]))