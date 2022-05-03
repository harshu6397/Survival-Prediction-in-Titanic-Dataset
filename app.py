import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction == 1:
        return render_template('index.html', prediction_text=f'The passanger is survived.')
    else:
        return render_template('index.html', prediction_text=f'The passanger is not survived.')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    if prediction == 1:
        return jsonify({"status" : "Survived"})
    else:
        return jsonify({"status" : "Not Survived"})

if __name__ == "__main__":
    app.run(debug=True)