from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)


def svm_model(sentence):
    load_model = joblib.load('svm.sav')
    return load_model.predict([sentence])[0]


@app.route('/result', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        sentence = request.form['text_to_analyse']
        result = svm_model(sentence)
        return render_template('index.html', result=result, sentence=sentence)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    redis_client = StrictRedis(host='redis', port=6379)
    app.debug = True
    app.run(host='0.0.0.0')
