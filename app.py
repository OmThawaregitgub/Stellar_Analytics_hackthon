from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import json
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'spaceml_secret_key_2024'

# ================= LOAD MODELS =================
print("="*50)
print("Loading models...")
print("="*50)

cls_model = joblib.load("models/classifier.pkl")
print(f"✅ Classification model loaded: {type(cls_model)}")
print(f"   Model classes: {cls_model.classes_}")

reg_model = joblib.load("models/regressor.pkl")
print(f"✅ Regression model loaded: {type(reg_model)}")

x_scaler = joblib.load("models/scalar_x.pkl")
print("✅ X_scaler loaded")

y_scaler = joblib.load("models/scalar_y.pkl")
print("✅ Y_scaler loaded")
print("="*50)

# ================= DATABASE INITIALIZATION =================
def init_db():
    try:
        conn = sqlite3.connect("cosmos.db")
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            label TEXT,
            confidence REAL,
            radius REAL,
            planet_type TEXT,
            koi_period REAL,
            koi_time0bk REAL,
            koi_impact REAL,
            koi_duration REAL,
            koi_depth REAL,
            koi_model_snr REAL,
            koi_steff REAL,
            koi_slogg REAL,
            koi_srad REAL,
            koi_teq REAL,
            koi_insol REAL
        )
        """)

        conn.commit()
        conn.close()
        print("✅ Database initialized: cosmos.db")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")

# Initialize database at startup
init_db()

# ================= SAVE TO DATABASE =================
def save_to_database(label, confidence, radius, planet_type, features_dict):
    try:
        conn = sqlite3.connect("cosmos.db")
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO predictions 
            (time, label, confidence, radius, planet_type,
             koi_period, koi_time0bk, koi_impact, koi_duration, koi_depth, 
             koi_model_snr, koi_steff, koi_slogg, koi_srad, koi_teq, koi_insol)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            label,
            float(confidence) if confidence else None,
            float(radius) if radius else None,
            planet_type,
            float(features_dict.get('koi_period', 0)),
            float(features_dict.get('koi_time0bk', 0)),
            float(features_dict.get('koi_impact', 0)),
            float(features_dict.get('koi_duration', 0)),
            float(features_dict.get('koi_depth', 0)),
            float(features_dict.get('koi_model_snr', 0)),
            float(features_dict.get('koi_steff', 0)),
            float(features_dict.get('koi_slogg', 0)),
            float(features_dict.get('koi_srad', 0)),
            float(features_dict.get('koi_teq', 0)),
            float(features_dict.get('koi_insol', 0)) if features_dict.get('koi_insol') else None
        ))

        # Keep only last 10 records
        cur.execute("SELECT COUNT(*) FROM predictions")
        count = cur.fetchone()[0]

        if count > 10:
            cur.execute("""
                DELETE FROM predictions 
                WHERE id IN (
                    SELECT id FROM predictions 
                    ORDER BY time ASC 
                    LIMIT ?
                )
            """, (count - 10,))

        conn.commit()
        conn.close()
        print(f"✅ Saved to database")
        return True
    except Exception as e:
        print(f"❌ Database save error: {e}")
        return False

# ================= EARTH RADIUS VISUALIZATION =================
def earth_radius_visualization(radius_earth_radii):
    """Create Earth radius visualization HTML/CSS"""
    earth_radius_km = 6371
    planet_radius_km = radius_earth_radii * earth_radius_km
    
    # Determine planet type based on radius
    if radius_earth_radii < 0.8:
        planet_type = "Sub-Earth"
        color = "#8c7853"
        description = "Small rocky world"
    elif radius_earth_radii < 1.25:
        planet_type = "Earth-like"
        color = "#4ade80"
        description = "Potentially habitable"
    elif radius_earth_radii < 2.0:
        planet_type = "Super-Earth"
        color = "#60a5fa"
        description = "Large rocky planet"
    elif radius_earth_radii < 4.0:
        planet_type = "Mini-Neptune"
        color = "#c084fc"
        description = "Gas-rich mini Neptune"
    elif radius_earth_radii < 10:
        planet_type = "Gas Giant"
        color = "#f59e0b"
        description = "Jupiter-like world"
    else:
        planet_type = "Unknown"
        color = "#94a3b8"
        description = "Unusual size"
    
    display_size = min(150, max(30, radius_earth_radii * 20))
    
    visualization = f"""
    <div class="radius-visualization" style="text-align: center; margin: 20px 0;">
        <div style="position: relative; width: 100%; display: flex; justify-content: center;">
            <div style="position: relative; width: 160px; height: 160px; margin: 0 auto;">
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                            width: 40px; height: 40px; border-radius: 50%; 
                            background: linear-gradient(135deg, #4ade80, #2563eb);
                            box-shadow: 0 0 20px rgba(74, 222, 128, 0.5);
                            border: 2px solid white; z-index: 1;">
                </div>
                <div style="position: absolute; top: 30px; left: 30px; color: white; font-size: 10px; opacity: 0.8;">Earth</div>
                
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                            width: {display_size}px; height: {display_size}px; border-radius: 50%; 
                            background: radial-gradient(circle at 30% 30%, {color}, {color}dd);
                            box-shadow: 0 0 30px {color};
                            border: 3px solid white;
                            z-index: 2;
                            animation: pulse 2s infinite;">
                </div>
                
                <svg width="200" height="200" style="position: absolute; top: -20px; left: -20px; z-index: 3; pointer-events: none;">
                    <line x1="20" y1="100" x2="180" y2="100" stroke="white" stroke-width="2" stroke-dasharray="5,5" opacity="0.5"/>
                    <line x1="100" y1="20" x2="100" y2="180" stroke="white" stroke-width="2" stroke-dasharray="5,5" opacity="0.5"/>
                </svg>
            </div>
        </div>
        
        <div style="margin-top: 30px; background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;">
            <h3 style="color: {color}; font-size: 2rem; margin-bottom: 5px;">{radius_earth_radii:.2f} R<sub>🜨</sub></h3>
            <p style="font-size: 1.1rem;">{planet_type}</p>
            <p style="opacity: 0.8; font-size: 0.9rem;">{description}</p>
            <p style="opacity: 0.7; margin-top: 10px;">≈ {planet_radius_km:,.0f} km</p>
        </div>
        
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

# ================= ROUTES =================

@app.route('/')
def home():
    return render_template('index.html', tab='home')

@app.route('/about')
def about():
    return render_template('index.html', tab='about')

@app.route('/history')
def history_page():
    return render_template('index.html', tab='history')

# ================= CLASSIFICATION ROUTE (GET + POST) =================
@app.route('/classification', methods=['GET', 'POST'])
def classification():
    result = None
    
    if request.method == 'POST':
        try:
            # Get form values - 9 for classification (NO teq, NO insol)
            koi_period = float(request.form["koi_period"])
            koi_time0bk = float(request.form["koi_time0bk"])
            koi_impact = float(request.form["koi_impact"])
            koi_duration = float(request.form["koi_duration"])
            koi_depth = float(request.form["koi_depth"])
            koi_model_snr = float(request.form["koi_model_snr"])
            koi_steff = float(request.form["koi_steff"])
            koi_slogg = float(request.form["koi_slogg"])
            koi_srad = float(request.form["koi_srad"])
            
            # Additional for regression
            koi_teq = float(request.form["koi_teq"])
            koi_insol = float(request.form["koi_insol"])

            print(f"✅ Form data received")

            # Store all features in dictionary
            features_dict = {
                'koi_period': koi_period,
                'koi_time0bk': koi_time0bk,
                'koi_impact': koi_impact,
                'koi_duration': koi_duration,
                'koi_depth': koi_depth,
                'koi_model_snr': koi_model_snr,
                'koi_steff': koi_steff,
                'koi_slogg': koi_slogg,
                'koi_srad': koi_srad,
                'koi_teq': koi_teq,
                'koi_insol': koi_insol
            }

            # ================= CLASSIFICATION =================
            # Features: [period, time0bk, impact, duration, depth, snr, teff, logg, srad]
            cls_features = np.array([
                koi_period, koi_time0bk, koi_impact, koi_duration, koi_depth,
                koi_model_snr, koi_steff, koi_slogg, koi_srad
            ]).reshape(1, -1)

            pred = cls_model.predict(cls_features)[0]
            print(f"✅ Classification Prediction: {pred}")

            try:
                probs = cls_model.predict_proba(cls_features)[0]
                confidence = round(max(probs) * 100, 1)
                probability = round(probs[1] * 100, 1)
                print(f"Class 0: {probs[0]*100:.1f}%, Class 1: {probs[1]*100:.1f}%")
            except:
                confidence = 50
                probability = 50

            # ================= REGRESSION =================
            # Features: [period, impact, duration, depth, snr, teff, logg, srad, teq, insol]
            reg_features = np.array([
                koi_period, koi_impact, koi_duration, koi_depth,
                koi_model_snr, koi_steff, koi_slogg, koi_srad,
                koi_teq, koi_insol
            ]).reshape(1, -1)

            reg_scaled = x_scaler.transform(reg_features)
            pred_scaled = reg_model.predict(reg_scaled)
            radius = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

            # Planet type based on radius
            if radius < 1:
                planet_type = "Earth-sized"
            elif radius < 4:
                planet_type = "Super-Earth"
            else:
                planet_type = "Gas Giant"

            # Determine if planet candidate (1 = CANDIDATE, 0 = FALSE POSITIVE)
            if pred == 1:
                label = "Confirmed Planet"
                result = {
                    "class": label,
                    "exists": True,
                    "confidence": confidence,
                    "probability": probability,
                    "radius": round(radius, 3),
                    "planet_type": planet_type
                }
                
                # Save to database
                save_to_database(
                    label="CONFIRMED",
                    confidence=confidence,
                    radius=round(radius, 3),
                    planet_type=planet_type,
                    features_dict=features_dict
                )
            else:
                label = "False Positive"
                result = {
                    "class": label,
                    "exists": False,
                    "confidence": confidence,
                    "probability": probability,
                    "radius": None,
                    "planet_type": None
                }
                
                # Save to database
                save_to_database(
                    label="FALSE POSITIVE",
                    confidence=confidence,
                    radius=None,
                    planet_type=None,
                    features_dict=features_dict
                )

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return render_template('index.html', tab='classification', error=str(e))
    
    return render_template('index.html', tab='classification', result=result)

@app.route('/get_history')
def get_history():
    """API endpoint to get history from database"""
    try:
        conn = sqlite3.connect("cosmos.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM predictions ORDER BY time DESC LIMIT 10")
        rows = cur.fetchall()

        history = []
        for row in rows:
            pred = {}
            for key in row.keys():
                val = row[key]
                if val is None:
                    pred[key] = None
                elif isinstance(val, (int, float)):
                    if key in ['confidence', 'radius']:
                        pred[key] = round(float(val), 2) if val else None
                    else:
                        pred[key] = val
                else:
                    pred[key] = str(val)
            history.append(pred)

        conn.close()
        return jsonify({"success": True, "history": history})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        conn = sqlite3.connect("cosmos.db")
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('classification'))

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)