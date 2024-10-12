from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from auto_modelo import TextPreprocessing

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

modelo = joblib.load('modelo_svm.pkl')

@app.route('/predict', methods=['OPTIONS', 'POST'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    if request.method == 'POST':
        data = request.get_json()
        df = pd.DataFrame(data)

        predictions = modelo.predict(df['TextosT'])
        probabilities = modelo.predict_proba(df['TextosT'])

        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
