import pandas as pd
import timeit
import numpy as np
from numpy import array
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn import linear_model, tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pickle

test = pd.read_csv('flask_input.csv')
X_new = test
X_new = X_new.astype(str)
    
le = preprocessing.LabelEncoder()
le.classes_ = np.load('classes.npy')
X_new_2 = le.fit_transform(X_new).reshape(1, -1)
    
loaded_model = pickle.load(open('reimburse_classifier.sav', 'rb'))
result = loaded_model.predict(X_new_2)

print(result)
