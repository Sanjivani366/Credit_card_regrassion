import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Predictor", page_icon="💳", layout="centered"
)

# Custom Styling (Dark Theme with your preferred gradient)
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #09090b, #dc2626);
        color: #f4f4f5;
    }
    .main-container {
        background: rgba(24, 24, 27, 0.9);
        padding: 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.7);
    }
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #f87171, #fca5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #a1a1aa;
        margin-bottom: 25px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained model safely
@st.cache_resource
def load_model():
  try:
    with open("CreditCardModel.pkl", "rb") as f:
      model = pickle.load(f)
    return model
  except Exception as e:
    return None


model = load_model()

# UI Layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1>Gradient Boosting Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Enter your details below to generate real-time"
    " inferences</p>",
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
  col1, col2 = st.columns(2)

  with col1:
    income = st.number_input("Income ($10k)", value=5.0, step=0.1)
    limit = st.number_input("Credit Limit", value=3000.0, step=100.0)
    rating = st.number_input("Credit Rating", value=250.0, step=10.0)
    cards = st.number_input(
        "Cards Held", min_value=1, max_value=10, value=2, step=1
    )
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)

  with col2:
    education = st.number_input(
        "Education (Years)", min_value=0, max_value=25, value=12, step=1
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    student = st.selectbox("Student", ["No", "Yes"])
    married = st.selectbox("Married", ["No", "Yes"])
    ethnicity = st.selectbox(
        "Ethnicity", ["Caucasian", "Asian", "African American"]
    )

  submit_button = st.form_submit_button(
      "Predict Outcome", use_container_width=True
  )

if submit_button:
  if model is None:
    st.error(
        "Model could not be loaded! Please check if CreditCardModel.pkl is in"
        " your GitHub repository."
    )
  else:
    # Encoding categories just like training
    gender_val = 1 if gender == "Female" else 0
    student_val = 1 if student == "Yes" else 0
    married_val = 1 if married == "Yes" else 0

    ethnicity_map = {"Caucasian": 0, "Asian": 1, "African American": 2}
    ethnicity_val = ethnicity_map.get(ethnicity, 0)

    features = np.array([[
        income,
        limit,
        rating,
        cards,
        age,
        education,
        gender_val,
        student_val,
        married_val,
        ethnicity_val,
    ]])

    try:
      prediction = float(model.predict(features)[0])
      st.success("Predicted Result Successfully Generated")
      st.metric(label="Predicted Value", value=f"{prediction:.2f}")
    except Exception as e:
      st.error(f"Error during prediction: {e}")

st.markdown("</div>", unsafe_allow_html=True)