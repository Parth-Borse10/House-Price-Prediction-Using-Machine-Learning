from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import json

app = Flask(__name__)
CORS(app)

# Global variables for the model and scaler
model = None
scaler = None
feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long']

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic housing data for demonstration"""
        np.random.seed(42)
        
        data = {
            'bedrooms': np.random.randint(1, 8, n_samples),
            'bathrooms': np.random.uniform(1, 4, n_samples),
            'sqft_living': np.random.randint(500, 5000, n_samples),
            'sqft_lot': np.random.randint(1000, 15000, n_samples),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'view': np.random.randint(0, 4, n_samples),
            'condition': np.random.randint(1, 6, n_samples),
            'grade': np.random.randint(1, 14, n_samples),
            'sqft_above': np.random.randint(500, 4000, n_samples),
            'sqft_basement': np.random.randint(0, 2000, n_samples),
            'yr_built': np.random.randint(1900, 2020, n_samples),
            'yr_renovated': np.random.randint(0, 2020, n_samples),
            'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005], n_samples),
            'lat': np.random.uniform(47.5, 47.8, n_samples),
            'long': np.random.uniform(-122.5, -121.8, n_samples)
        }
        
        # Generate price based on features with some noise
        base_price = (
            data['sqft_living'] * 200 +
            data['bedrooms'] * 50000 +
            data['bathrooms'] * 40000 +
            data['grade'] * 10000 +
            data['waterfront'] * 300000 +
            data['view'] * 25000
        )
        
        # Add noise and ensure minimum price
        data['price'] = np.maximum(base_price + np.random.normal(0, 100000, n_samples), 50000)
        
        return pd.DataFrame(data)
    
    def train_model(self, df):
        """Train the Random Forest model"""
        # Prepare features and target
        X = df[feature_names]
        y = df['price']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        
        return {
            'mae': mae,
            'r2': r2,
            'feature_importance': dict(zip(feature_names, self.model.feature_importances_))
        }
    
    def predict_price(self, features):
        """Predict house price for given features"""
        if not self.is_trained:
            return None
        
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        prediction = self.model.predict(feature_df)[0]
        
        return prediction

# Initialize the predictor
predictor = HousePricePredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train the machine learning model"""
    try:
        # Generate sample data and train model
        df = predictor.generate_sample_data(1000)
        metrics = predictor.train_model(df)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully!',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict house price based on input features"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = feature_names
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'message': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Convert features to appropriate types
        features = {}
        for field in required_fields:
            try:
                features[field] = float(data[field])
            except ValueError:
                return jsonify({
                    'success': False,
                    'message': f'Invalid value for {field}'
                }), 400
        
        # Make prediction
        if not predictor.is_trained:
            return jsonify({
                'success': False,
                'message': 'Model not trained yet. Please train the model first.'
            }), 400
        
        predicted_price = predictor.predict_price(features)
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'features': features
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get the list of required features"""
    return jsonify({
        'features': feature_names,
        'descriptions': {
            'bedrooms': 'Number of bedrooms',
            'bathrooms': 'Number of bathrooms',
            'sqft_living': 'Square footage of living area',
            'sqft_lot': 'Square footage of lot',
            'floors': 'Number of floors',
            'waterfront': 'Waterfront view (0=No, 1=Yes)',
            'view': 'Quality of view (0-4)',
            'condition': 'Overall condition (1-5)',
            'grade': 'Construction quality (1-13)',
            'sqft_above': 'Square footage above ground',
            'sqft_basement': 'Square footage of basement',
            'yr_built': 'Year built',
            'yr_renovated': 'Year renovated (0 if not)',
            'zipcode': 'Zipcode area',
            'lat': 'Latitude',
            'long': 'Longitude'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)