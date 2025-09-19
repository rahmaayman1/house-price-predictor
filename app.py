from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

def load_model():
    
    print("Current directory:", os.getcwd())
    print("Files in directory:", os.listdir("."))
    
    possible_names = [
        "xgb_house_price_model.pkl",
        "./xgb_house_price_model.pkl", 
        "best_xgb.pkl",
        "xgboost_model.pkl"
    ]
    
    for name in possible_names:
        try:
            model = joblib.load(name)
            print(f"✅ Model loaded successfully from: {name}")
            return model
        except Exception as e:
            print(f"❌ Failed to load {name}: {e}")
    
    print("❌ Could not load any model file")
    return None

model = load_model()

REQUIRED_FEATURES = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
    'waterfront', 'view', 'condition', 'grade', 'sqft_basement',
    'yr_renovated', 'zipcode', 'lat', 'long', 'year', 'has_basement',
    'house_age', 'is_renovated', 'years_since_renovation', 'sqft_lot15_log',
    'basement_ratio', 'price_per_sqft', 'living_to_lot_ratio', 'basement_to_lot_ratio'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check model file.'}), 500
        
        data = request.json if request.is_json else request.form.to_dict()
        
        bedrooms = int(data.get('bedrooms', 3))
        bathrooms = int(data.get('bathrooms', 2))
        sqft_living = int(data.get('sqft_living', 2000))
        year_built = int(data.get('year_built', 1990))
        grade = int(data.get('grade', 7)) 
        

        if not (1 <= bedrooms <= 20):
            return jsonify({'error': 'Bedrooms must be between 1 and 20'}), 400
        if not (1 <= bathrooms <= 20):
            return jsonify({'error': 'Bathrooms must be between 1 and 20'}), 400
        if not (100 <= sqft_living <= 50000):
            return jsonify({'error': 'Living area must be between 100 and 50,000 sqft'}), 400
        if not (1800 <= year_built <= 2024):
            return jsonify({'error': 'Year built must be between 1800 and 2024'}), 400
        if not (1 <= grade <= 13):
            return jsonify({'error': 'Grade must be between 1 and 13'}), 400
        
        
        sqft_lot_value = 7500.0
        sqft_basement_value = 0.0
        
        input_data = {
            'bedrooms': float(bedrooms),
            'bathrooms': float(bathrooms),
            'sqft_living': float(sqft_living),
            'sqft_lot': sqft_lot_value,      
            'floors': 1.5,                   
            'waterfront': 0,                
            'view': 0,                       
            'condition': 3,                  
            'grade': float(grade),           
            'sqft_basement': sqft_basement_value,  
            'yr_renovated': 0,               
            'zipcode': 98178,                
            'lat': 47.5112,                  
            'long': -122.257,                
            'year': float(year_built),       
            'has_basement': 1 if sqft_basement_value > 0 else 0,  
            'house_age': float(2024 - year_built),  
            'is_renovated': 0,               
            'years_since_renovation': 0,     
            'sqft_lot15_log': np.log1p(sqft_lot_value),  
            
            
            'basement_ratio': sqft_basement_value / sqft_living if sqft_living > 0 else 0.0,
            'price_per_sqft': 150.0,         
            'living_to_lot_ratio': sqft_living / sqft_lot_value if sqft_lot_value > 0 else 0.0,
            'basement_to_lot_ratio': sqft_basement_value / sqft_lot_value if sqft_lot_value > 0 else 0.0
        }
        
        
        df = pd.DataFrame([input_data])
        
        df = df[REQUIRED_FEATURES]
        
        print(f"Input data shape: {df.shape}")
        print(f"Features: {list(df.columns)}")
        
        # prediction
        prediction = model.predict(df)[0]
        
        
        if prediction < 0:
            prediction = abs(prediction)
        
        formatted_price = f"${prediction:,.2f}"
        
        return jsonify({
            'success': True,
            'predicted_price': formatted_price,
            'price_value': float(prediction),
            'input_summary': {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft_living': sqft_living,
                'year_built': year_built,
                'grade': grade,
                'house_age': 2024 - year_built
            }
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")  
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 400

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(REQUIRED_FEATURES)
    })


@app.route('/debug')
def debug():
    return jsonify({
        'current_dir': os.getcwd(),
        'files': os.listdir("."),
        'model_loaded': model is not None,
        'model_type': str(type(model)) if model else None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)