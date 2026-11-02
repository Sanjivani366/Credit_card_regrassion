import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained GradientBoostingRegressor model
try:
    with open('CreditCardModel.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction Portal</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #09090b, #dc2626);
            --card-bg: rgba(24, 24, 27, 0.9);
            --text-color: #f4f4f5;
            --accent-color: #ef4444;
            --accent-hover: #dc2626;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        body {
            background: var(--bg-gradient);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
            width: 100%;
            max-width: 700px;
            padding: 40px;
            margin: 20px 0;
        }
        h1 {
            text-align: center;
            margin-bottom: 8px;
            font-size: 2.2rem;
            background: linear-gradient(90deg, #f87171, #fca5a5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #a1a1aa;
            margin-bottom: 30px;
            font-size: 0.95rem;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #d4d4d8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        input, select {
            background: rgba(9, 9, 11, 0.8);
            border: 1px solid #3f3f46;
            border-radius: 8px;
            padding: 12px;
            color: #fff;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.25);
        }
        button {
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 14px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.3s ease, transform 0.2s ease;
            margin-top: 10px;
            grid-column: span 2;
        }
        button:hover {
            background: var(--accent-hover);
            transform: translateY(-2px);
        }
        .result-card {
            margin-top: 30px;
            background: rgba(9, 9, 11, 0.95);
            border: 2px solid #22c55e;
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
        }
        .result-card h2 {
            font-size: 1.1rem;
            color: #4ade80;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .prediction-value {
            font-size: 2.8rem;
            font-weight: 700;
            color: #ffffff;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            button, .form-group {
                grid-column: span 1;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Gradient Boosting Predictor</h1>
        <p class="subtitle">Enter your details below to generate real-time inferences</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="income">Income ($10k)</label>
                    <input type="number" step="any" id="income" name="income" required value="{{ request.form.get('income', '') }}">
                </div>
                
                <div class="form-group">
                    <label for="limit">Credit Limit</label>
                    <input type="number" step="any" id="limit" name="limit" required value="{{ request.form.get('limit', '') }}">
                </div>

                <div class="form-group">
                    <label for="rating">Credit Rating</label>
                    <input type="number" step="any" id="rating" name="rating" required value="{{ request.form.get('rating', '') }}">
                </div>

                <div class="form-group">
                    <label for="cards">Cards Held</label>
                    <input type="number" id="cards" name="cards" required value="{{ request.form.get('cards', '') }}">
                </div>

                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" required value="{{ request.form.get('age', '') }}">
                </div>

                <div class="form-group">
                    <label for="education">Education (Years)</label>
                    <input type="number" id="education" name="education" required value="{{ request.form.get('education', '') }}">
                </div>

                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select id="gender" name="gender" required>
                        <option value="Male" {% if request.form.get('gender') == 'Male' %}selected{% endif %}>Male</option>
                        <option value="Female" {% if request.form.get('gender') == 'Female' %}selected{% endif %}>Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="student">Student</label>
                    <select id="student" name="student" required>
                        <option value="No" {% if request.form.get('student') == 'No' %}selected{% endif %}>No</option>
                        <option value="Yes" {% if request.form.get('student') == 'Yes' %}selected{% endif %}>Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="married">Married</label>
                    <select id="married" name="married" required>
                        <option value="No" {% if request.form.get('married') == 'No' %}selected{% endif %}>No</option>
                        <option value="Yes" {% if request.form.get('married') == 'Yes' %}selected{% endif %}>Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="ethnicity">Ethnicity</label>
                    <select id="ethnicity" name="ethnicity" required>
                        <option value="Caucasian" {% if request.form.get('ethnicity') == 'Caucasian' %}selected{% endif %}>Caucasian</option>
                        <option value="Asian" {% if request.form.get('ethnicity') == 'Asian' %}selected{% endif %}>Asian</option>
                        <option value="African American" {% if request.form.get('ethnicity') == 'African American' %}selected{% endif %}>African American</option>
                    </select>
                </div>

                <button type="submit">Predict Outcome</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h2>Predicted Result Successfully Generated</h2>
            <div class="prediction-value">{{ "%.2f"|format(prediction) }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            income = float(request.form['income'])
            limit = float(request.form['limit'])
            rating = float(request.form['rating'])
            cards = float(request.form['cards'])
            age = float(request.form['age'])
            education = float(request.form['education'])
            
            gender_val = 1 if request.form['gender'] == 'Female' else 0
            student_val = 1 if request.form['student'] == 'Yes' else 0
            married_val = 1 if request.form['married'] == 'Yes' else 0
            
            ethnicity_map = {'Caucasian': 0, 'Asian': 1, 'African American': 2}
            ethnicity_val = ethnicity_map.get(request.form['ethnicity'], 0)

            features = np.array([[
                income, limit, rating, cards, age, 
                education, gender_val, student_val, married_val, ethnicity_val
            ]])
            
            if model:
                prediction = float(model.predict(features)[0])
                print(f"Prediction calculated: {prediction}")
        except Exception as e:
            print(f"Prediction inference error: {e}")
            prediction = None

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)