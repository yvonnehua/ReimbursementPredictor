from __future__ import print_function
import sys
from flask import Flask, render_template, request, redirect
import os
from werkzeug import secure_filename
import pandas as pd
import numpy as np
from numpy import array
import sklearn
from sklearn import linear_model, tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/upload', methods = ['GET', 'POST'])
def upload_file2():
   if request.method == 'POST':
      f = request.files['file']
      os.makedirs(os.path.join(app.instance_path, 'htmlfi'), exist_ok=True)
      f.save(os.path.join(app.instance_path, 'htmlfi', secure_filename(f.filename)))
      print(f)      

      X_new = pd.read_csv('flask_input.csv', index_col=0, dtype = object, header = None)
      X_new = X_new.astype(str)
    
      le = preprocessing.LabelEncoder()
      le.classes_ = np.load('classes.npy')
      X_new_2 = le.fit_transform(X_new).reshape(1, -1)
    
      loaded_model = pickle.load(open('reimburse_classifier.sav', 'rb'))
      result = loaded_model.predict(X_new_2)
      return str(result[0])

		
if __name__ == '__main__':
   app.run(debug = True)
