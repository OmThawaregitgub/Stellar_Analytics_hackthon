from flask import Flask, render_template, request, session
import numpy as np
import pickle
import os
import math

app = Flask(__name__)
app.secret_key = 'spaceml_secret_key_2024'

# Load models and scalers
MODELS_PATH = 'models'

# Classification model (no scaling needed)
with open(os.path.join(MODELS_PATH, 'classification_model.pkl'), 'rb') as f:
    classification_model = pickle.load(f)

# Regression model and scalers
with open(os.path.join(MODELS_PATH, 'regression_model.pkl'), 'rb') as f:
    regression_model = pickle.load(f)

with open(os.path.join(MODELS_PATH, 'x_scaler.pkl'), 'rb') as f:
    x_scaler = pickle.load(f)

with open(os.path.join(MODELS_PATH, 'y_scaler.pkl'), 'rb') as f:
    y_scaler = pickle.load(f)

# Classification features (no scaling required)
CLASS_FEATURES = [
    'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad', 'koi_teq'
]

# Regression features (need scaling)
REG_FEATURES = [
    'koi_period', 'koi_impact', 'koi_duration', 'koi_depth',
    'koi_model_snr', 'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_teq', 'koi_insol'
]

def earth_radius_visualization(radius_earth_radii):
    """
    Create Earth radius visualization HTML/CSS
    radius_earth_radii: planet radius in Earth radii units
    """
    # Earth radius in km (for reference)
    earth_radius_km = 6371
    
    # Calculate planet radius in km
    planet_radius_km = radius_earth_radii * earth_radius_km
    
    # Determine planet type based on radius
    if radius_earth_radii < 0.8:
        planet_type = "Sub-Earth"
        color = "#8c7853"  # rocky brown
        description = "Small rocky world"
    elif radius_earth_radii < 1.25:
        planet_type = "Earth-like"
        color = "#4ade80"  # earth green
        description = "Potentially habitable"
    elif radius_earth_radii < 2.0:
        planet_type = "Super-Earth"
        color = "#60a5fa"  # blue
        description = "Large rocky planet"
    elif radius_earth_radii < 4.0:
        planet_type = "Mini-Neptune"
        color = "#c084fc"  # purple
        description = "Gas-rich mini Neptune"
    elif radius_earth_radii < 10:
        planet_type = "Gas Giant"
        color = "#f59e0b"  # orange
        description = "Jupiter-like world"
    else:
        planet_type = "Unknown"
        color = "#94a3b8"  # gray
        description = "Unusual size"
    
    # Size for visualization (max 150px)
    display_size = min(150, max(30, radius_earth_radii * 20))
    
    # Create SVG visualization
    visualization = f"""
    <div class="radius-visualization" style="text-align: center; margin: 20px 0;">
        <div style="position: relative; width: 100%; display: flex; justify-content: center;">
            <!-- Earth reference (small) -->
            <div style="position: relative; width: 160px; height: 160px; margin: 0 auto;">
                <!-- Earth outline (for scale) -->
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                            width: 40px; height: 40px; border-radius: 50%; 
                            background: linear-gradient(135deg, #4ade80, #2563eb);
                            box-shadow: 0 0 20px rgba(74, 222, 128, 0.5);
                            border: 2px solid white; z-index: 1;">
                </div>
                <!-- Earth label -->
                <div style="position: absolute; top: 30px; left: 30px; color: white; font-size: 10px; opacity: 0.8;">Earth</div>
                
                <!-- Discovered planet -->
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                            width: {display_size}px; height: {display_size}px; border-radius: 50%; 
                            background: radial-gradient(circle at 30% 30%, {color}, {color}dd);
                            box-shadow: 0 0 30px {color};
                            border: 3px solid white;
                            z-index: 2;
                            animation: pulse 2s infinite;">
                </div>
                
                <!-- Size indicator lines -->
                <svg width="200" height="200" style="position: absolute; top: -20px; left: -20px; z-index: 3; pointer-events: none;">
                    <line x1="20" y1="100" x2="180" y2="100" stroke="white" stroke-width="2" stroke-dasharray="5,5" opacity="0.5"/>
                    <line x1="100" y1="20" x2="100" y2="180" stroke="white" stroke-width="2" stroke-dasharray="5,5" opacity="0.5"/>
                </svg>
            </div>
        </div>
        
        <!-- Stats -->
        <div style="margin-top: 30px; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;">
            <h3 style="color: {color}; font-size: 2rem; margin-bottom: 5px;">{radius_earth_radii:.2f} R<sub>🜨</sub></h3>
            <p style="font-size: 1.1rem;">{planet_type}</p>
            <p style="opacity: 0.8; font-size: 0.9rem;">{description}</p>
            <p style="opacity: 0.7; margin-top: 10px;">≈ {planet_radius_km:,.0f} km</p>
        </div>
        
        <!-- Size comparison text -->
        <div style="margin-top: 15px; font-size: 0.9rem; opacity: 0.7;">
            {radius_earth_radii:.2f} × Earth radius
        </div>
    </div>
    
    <style>
        @keyframes pulse {{
            0% {{ transform: translate(-50%, -50%) scale(1); opacity: 1; }}
            50% {{ transform: translate(-50%, -50%) scale(1.05); opacity: 0.9; }}
            100% {{ transform: translate(-50%, -50%) scale(1); opacity: 1; }}
        }}
    </style>
    """
    
    return visualization

@app.route('/')
def home():
    return render_template('index.html', tab='home')

@app.route('/about')
def about():
    return render_template('index.html', tab='about')

@app.route('/analysis')
def analysis():
    # Clear session when starting new analysis
    session.clear()
    return render_template('index.html', tab='analysis')

@app.route('/history')
def history():
    # Just render the template with history tab active
    return render_template('index.html', tab='history')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Get form data
        features = []
        feature_dict = {}
        for feature in CLASS_FEATURES:
            value = request.form.get(feature)
            if not value:
                return render_template('index.html', tab='analysis', 
                                      message=f"Missing value for {feature}")
            features.append(float(value))
            feature_dict[feature] = float(value)
        
        # Convert to numpy array and reshape
        X = np.array(features).reshape(1, -1)
        
        # Make prediction - get the raw prediction value
        prediction = classification_model.predict(X)[0]
        
        # Convert numpy types to Python native types
        prediction = int(prediction)
        
        # Store features in session for regression if needed
        session['classification_features'] = [float(f) for f in features]
        session['feature_dict'] = feature_dict
        session['prediction'] = prediction
        
        # 0 = CANDIDATE, 1 = FALSE POSITIVE (based on debug output)
        is_planet = prediction == 0
        
        if is_planet:
            # Show regression form
            return render_template('index.html', tab='analysis', 
                                 show_reg_form=True,
                                 result="✅ PLANET CANDIDATE DETECTED",
                                 features=feature_dict)
        else:
            # For false positives, show message and DO NOT show regression form
            return render_template('index.html', tab='analysis',
                                 result="❌ FALSE POSITIVE",
                                 message="No planet detected. This is likely a false positive. Try different parameters.",
                                 features=feature_dict)
    
    except Exception as e:
        return render_template('index.html', tab='analysis',
                             message=f"Error: {str(e)}")

@app.route('/regress', methods=['POST'])
def regress():
    try:
        # Check if this is a valid planet candidate
        if 'prediction' not in session or session['prediction'] != 0:
            return render_template('index.html', tab='analysis',
                                 message="Invalid session. Only confirmed planets can calculate radius.")
        
        # Get insol from form
        koi_insol = float(request.form.get('koi_insol'))
        
        # Retrieve classification features from session
        if 'classification_features' not in session:
            return render_template('index.html', tab='analysis',
                                 message="Session expired. Please restart analysis.")
        
        features = session['classification_features']
        feature_dict = session.get('feature_dict', {})
        feature_dict['koi_insol'] = koi_insol
        
        # Add insol to features
        features.append(koi_insol)
        
        # Convert to numpy array
        X = np.array(features).reshape(1, -1)
        
        # Scale features
        X_scaled = x_scaler.transform(X)
        
        # Make prediction
        y_pred_scaled = regression_model.predict(X_scaled)
        
        # Inverse transform to get actual koi_prad
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        
        # Convert to Python float
        y_pred = float(y_pred)
        
        # Generate visualization
        visualization = earth_radius_visualization(y_pred)
        
        return render_template('index.html', tab='analysis',
                             result=f"🌍 {y_pred:.2f} R🜨",
                             visualization=visualization,
                             show_visualization=True,
                             features=feature_dict)
    
    except Exception as e:
        return render_template('index.html', tab='analysis',
                             message=f"Error: {str(e)}")

@app.route('/reset')
def reset():
    session.clear()
    return render_template('index.html', tab='analysis')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
