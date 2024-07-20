import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocessing import preprocess_text
from flask import Flask, request, jsonify
import joblib
from googletrans import Translator

app = Flask(__name__)

# Charger le modèle
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl'))
model = joblib.load(model_path)

# Initialiser le traducteur
translator = Translator()

# Mapping des prédictions
LABELS = {
    0: "Fiable",
    1: "Peu Fiable"
}

@app.route('/')
def home():
    return "Server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        title = data.get('title', '')
        author = data.get('author', '')
        text = data.get('text', '')

        # Traduction du texte en anglais si nécessaire
        if not text.strip().startswith('en:'):
            translation = translator.translate(text, src='fr', dest='en')
            text = translation.text

        # Prétraitement du texte
        processed_text = preprocess_text(text)

        # Prédiction
        prediction = model.predict([processed_text])[0]

        # Retourner la réponse en JSON avec les labels lisibles
        return jsonify({'prediction': LABELS.get(int(prediction), "Inconnu")}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

