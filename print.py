from __future__ import print_function
import sys
from flask import Flask, render_template, request, redirect
import os
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/button')
def button_clicked():
    print('Hello world!', file=sys.stderr)
    return redirect('/')

if __name__ == '__main__':
   app.run(debug = True)
