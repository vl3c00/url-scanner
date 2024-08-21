from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def sanitization(web):
    web = web.lower()
    token = []
    dot_token_slash = []
    raw_slash = str(web).split('/')
    for i in raw_slash:
        raw1 = str(i).split('-')
        slash_token = []
        for j in range(0, len(raw1)):
            raw2 = str(raw1[j]).split('.')
            slash_token = slash_token + raw2
        dot_token_slash = dot_token_slash + raw1 + slash_token
    token = list(set(dot_token_slash))
    if 'com' in token:
        token.remove('com')
    return token

def load_model_and_vectorizer():
    try:
        # Load model
        with open("Classifier/pickel_model.pkl", 'rb') as f1:
            lgr = pickle.load(f1)
        print("Model loaded successfully.")
        
        # Load vectorizer
        with open("Classifier/pickel_vector.pkl", 'rb') as f2:
            vectorizer = pickle.load(f2)
        print("Vectorizer loaded successfully.")
        
        return lgr, vectorizer
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

@app.route('/check-url', methods=['POST'])
def check_url():
    data = request.json
    url = data.get('url', '')
    
    if not model or not vectorizer:
        return jsonify({'error': 'Model or vectorizer not loaded properly.'}), 500
    
    # Sanitize the URL
    sanitized_url = sanitization(url)
    
    # Using whitelist filter
    whitelist = ['hackthebox.eu', 'root-me.org', 'gmail.com']
    if url in whitelist:
        return jsonify({'prediction': 'good'})
    
    try:
        # Predict
        x = vectorizer.transform([url])
        y_predict = model.predict(x)
        prediction = y_predict[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        print(f"Error predicting URL: {e}")
        return jsonify({'error': 'Prediction failed.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
