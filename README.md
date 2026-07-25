# 🧠 AI IQ Predictor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-ai--iq--predictor.vercel.app-black?style=for-the-badge&logo=vercel)](https://ai-iq-predictor.vercel.app/)

A full-stack web application that predicts your IQ score based on your CGPA using a machine learning model trained on real student data. Built with a premium Panera-inspired UI and powered by a Flask + scikit-learn backend.

🔗 **Live:** [https://ai-iq-predictor.vercel.app/](https://ai-iq-predictor.vercel.app/)

---

## 📸 Preview

> Enter your CGPA → Click **Predict IQ** → Instantly see your predicted IQ, cognitive category, and skill profile.

---

## 🗂️ Project Structure

```
AI IQ Predictor/
├── api/
│   └── index.py                # Vercel serverless entry point
├── backend/
│   ├── backend.py              # Local dev Flask server
│   └── templates/
│       └── index.html          # Frontend UI (served by Flask)
├── dataset/
│   └── student_clustering.csv  # Training dataset (200 student records)
├── requirements.txt            # Python dependencies for Vercel
├── vercel.json                 # Vercel routing + Python runtime config
└── README.md
```

---

## ⚙️ How It Works

1. **Dataset** — 200 student records with `cgpa` (0–10) and `iq` columns.
2. **Model** — A `LinearRegression` model is trained on startup:
   - Slope: ~4.0 (each CGPA point ≈ +4 IQ points)
   - Intercept: ~74
   - R² = 0.29, MAE = 9.74, RMSE = 10.25
3. **API** — The Flask server exposes a `/predict` endpoint.
4. **Frontend** — The browser POSTs the CGPA value and renders the ML result with a count-up animation.

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install flask flask-cors scikit-learn pandas numpy
```

### 2. Run the server

```bash
cd backend
python backend.py
```

### 3. Open in browser

```
http://127.0.0.1:5001
```

---

## 🔌 API Reference

### `POST /predict`

Predict IQ from a given CGPA.

**Request**
```json
{ "cgpa": 8.45 }
```

**Response**
```json
{
  "iq": 108,
  "cgpa": 8.45,
  "category": "Average",
  "description": "Your score reflects steady cognitive ability and balanced reasoning.",
  "skills": {
    "logical": "Moderate",
    "problem": "Average",
    "learning": "Average"
  }
}
```

### `GET /health`

Returns server and model status.

```json
{ "status": "ok", "model": "LinearRegression", "r2": 0.2865 }
```

---

## 🎨 IQ Categories

| IQ Range | Category       | Logical Thinking | Problem Solving | Learning Ability |
|----------|---------------|------------------|-----------------|------------------|
| 140+     | Genius         | Exceptional      | Mastery         | Elite            |
| 130–139  | Very Superior  | Exceptional      | Masterful       | Elite            |
| 120–129  | Above Average  | High             | Strong          | Excellent        |
| 110–119  | High Average   | Solid            | Capable         | Good             |
| 100–109  | Average        | Moderate         | Average         | Average          |
| 90–99    | Low Average    | Developing       | Basic           | Moderate         |
| < 90     | Below Average  | Growing          | Developing      | Building         |

---

## 🛠️ Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Frontend  | HTML, CSS, Vanilla JavaScript     |
| Backend   | Python, Flask, Flask-CORS         |
| ML Model  | scikit-learn (LinearRegression)   |
| Data      | pandas, NumPy                     |
| Fonts     | Google Fonts (Playfair Display, Inter) |

---

## ☁️ Deployment

The app is deployed on **Vercel** using the Python serverless runtime.

🔗 **Live URL:** [https://ai-iq-predictor.vercel.app/](https://ai-iq-predictor.vercel.app/)

| File | Role |
|---|---|
| `vercel.json` | Routes all traffic to `api/index.py`, sets Python runtime |
| `requirements.txt` | Installed by Vercel on each build |
| `api/index.py` | Serverless Flask app — same logic as `backend.py`, paths adjusted for Vercel |

To redeploy, simply push to `main` — Vercel auto-deploys on every commit.

---

## 📄 License

MIT — feel free to fork, modify, and use.
