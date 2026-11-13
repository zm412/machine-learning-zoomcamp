import pickle

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

from model import normalize_record


model_file = 'pipeline.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    survey = request.get_json()

    record = normalize_record([survey])

    X = dv.transform(record)
    y_pred = model.predict_proba(X)[0, 1]
    result = y_pred >= 0.5

    result = {
        'hire_probability': float(y_pred),
        'hire': bool(result)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
