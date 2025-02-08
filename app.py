from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import requests  # Keep using synchronous requests
import concurrent.futures  # To make scraper API calls non-blocking

# Load trained model and scaler
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

# Your Insta-scraper API URL
SCRAPER_API_URL = "http://127.0.0.1:3100/scrape"  # Update with actual API URL

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

executor = concurrent.futures.ThreadPoolExecutor()  # Thread pool for async behavior

def fetch_scraper_data(profile_url):
    """Function to call the scraper API (runs in a separate thread)."""
    response = requests.post(SCRAPER_API_URL, json={"profile": profile_url})
    if response.status_code == 200:
        return response.json()
    return None




@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        profile_url = data.get("profile")

        if not profile_url:
            return jsonify({"error": "Missing 'profile' in request"}), 400

        # Run scraper API call in a separate thread
        future = executor.submit(fetch_scraper_data, profile_url)
        scraper_data = future.result()  # Wait for the result

        if not scraper_data:
            return jsonify({"error": "Failed to fetch data from scraper"}), 500

        print(scraper_data)
        # Extract features from the scraper response
        features = np.array([scraper_data])  # Expecting a list of features

        # Scale input data
        features_scaled = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(features_scaled)[0]  # Get first output

        # Return JSON response
        return jsonify({"fake_score": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Normal synchronous Flask run
