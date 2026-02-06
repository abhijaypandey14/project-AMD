from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
app = Flask(__name__)
def train_model():
    """
    Trains a Random Forest model using synthetic data.
    Corrects the 'bitwise' error by using proper parentheses.
    """
    np.random.seed(42)
    n_samples = 1000
    hours = np.random.randint(0, 24, n_samples)
    weather = np.random.randint(0, 2, n_samples)
    density = (
        ((hours >= 8) & (hours <= 10)) * 30 + 
        ((hours >= 17) & (hours <= 19)) * 40 +
        (weather * 15) + 
        np.random.normal(10, 5, n_samples)
    )
    density = np.clip(density, 0, 100)
    df = pd.DataFrame({'hour': hours, 'weather': weather, 'density': density})
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(df[['hour', 'weather']], df['density'])
    return model
traffic_model = train_model()
@app.route('/')
def home():
    """Serves the main website page."""
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives data from the website and returns an AI traffic prediction.
    """
    data = request.json
    hour = int(data.get('hour', 12))
    weather = int(data.get('weather', 0))
    prediction = traffic_model.predict([[hour, weather]])[0]
    return jsonify({
        'predicted_density': round(prediction, 2),
        'status': "Heavy" if prediction > 60 else "Moderate" if prediction > 30 else "Light"
    })
if __name__ == '__main__':
    app.run(debug=True)