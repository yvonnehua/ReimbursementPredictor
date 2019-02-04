from __future__ import print_function
import sys
from flask import Flask, render_template, request, redirect
import os
from werkzeug import secure_filename

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
      return 'file uploaded successfully'

		
if __name__ == '__main__':
   app.run(debug = True)
