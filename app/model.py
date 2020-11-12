from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)


def svm_model(sentence):
    load_model = joblib.load('svm.sav')
    return load_model.predict([sentence])


@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sentence = request.form['Name']  # name of the insert from html
        # blob = TextBlob(sentence)
        result = svm_model(sentence)
        return render_template('index.html',
                               result=result)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.debug = True
    app.run()
