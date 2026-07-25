import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths (relative to this file, works both locally and on Vercel) ────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR  = os.path.join(BASE_DIR, '..', 'backend', 'templates')
DATA_PATH     = os.path.join(BASE_DIR, '..', 'dataset', 'student_clustering.csv')

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# ── Train model on cold start ──────────────────────────────────────────────────
data = pd.read_csv(DATA_PATH)
X = data[['cgpa']]
y = data['iq']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_skills(iq: float) -> dict:
    if iq >= 140: return {"category": "Genius",       "logical": "Exceptional", "problem": "Mastery",    "learning": "Elite"}
    if iq >= 130: return {"category": "Very Superior","logical": "Exceptional", "problem": "Masterful",  "learning": "Elite"}
    if iq >= 120: return {"category": "Above Average","logical": "High",        "problem": "Strong",     "learning": "Excellent"}
    if iq >= 110: return {"category": "High Average", "logical": "Solid",       "problem": "Capable",    "learning": "Good"}
    if iq >= 100: return {"category": "Average",      "logical": "Moderate",    "problem": "Average",    "learning": "Average"}
    if iq >= 90:  return {"category": "Low Average",  "logical": "Developing",  "problem": "Basic",      "learning": "Moderate"}
    return               {"category": "Below Average","logical": "Growing",     "problem": "Developing", "learning": "Building"}


def get_description(iq: float) -> str:
    if iq >= 130: return "Your score reflects outstanding analytical power and rapid learning."
    if iq >= 120: return "Your score indicates strong cognitive potential and analytical ability."
    if iq >= 110: return "Your score reflects solid reasoning skills and above-average aptitude."
    if iq >= 100: return "Your score reflects steady cognitive ability and balanced reasoning."
    if iq >= 90:  return "Your score shows baseline reasoning and everyday problem-solving skills."
    return "Keep working on your skills — consistent effort leads to growth!"


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json(force=True)
        cgpa = float(body.get('cgpa', 0))
        if cgpa < 0 or cgpa > 10:
            return jsonify({"error": "CGPA must be between 0 and 10"}), 400

        raw_iq = model.predict(pd.DataFrame([[cgpa]], columns=['cgpa']))[0]
        iq = int(round(max(70, min(145, raw_iq))))

        skills = get_skills(iq)
        return jsonify({
            "iq":          iq,
            "cgpa":        cgpa,
            "category":    skills["category"],
            "description": get_description(iq),
            "skills": {
                "logical":  skills["logical"],
                "problem":  skills["problem"],
                "learning": skills["learning"]
            }
        })
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model": "LinearRegression", "r2": round(r2, 4)})


# ── Local dev entry point (not used by Vercel) ─────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5001)
