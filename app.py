import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# 1. Load dataset
df = pd.read_csv("country_2026_risk_profile-selected-columns.csv")

# 2. Features and target
X = df[["avg_screen_time", "avg_social_media_hours", 
        "avg_sleep_hours", "avg_addiction_score"]]
y = df["avg_mental_health_risk"]

# 3. Train model (or load saved one)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Optional: Save model for reuse
joblib.dump(model, "mental_health.pkl")

# Risk categorization function
def categorize_risk(score):
    if score < 37.0:
        return "🟢 Low Risk"
    elif 37.0 <= score < 39.0:
        return "🟡 Moderate Risk"
    else:
        return "🔴 High Risk"

# 4. Streamlit UI
st.title("🌍 Country Mental Health Risk Predictor")
st.write("Enter lifestyle averages to predict mental health risk:")

# Input sliders
screen_time = st.slider("📱 Avg Screen Time (hours)", 3.0, 12.0, 5.0)
social_media = st.slider("💬 Avg Social Media Hours", 1.0, 12.0, 2.7)
sleep_hours = st.slider("😴 Avg Sleep Hours", 1.0, 12.0, 7.5)
addiction_score = st.slider("🎮 Avg Addiction Score", 40.0, 60.0, 51.0)

# Predict button (unique key to avoid duplicate error)
if st.button("🔮 Predict Risk", key="predict_button"):
    new_data = [[screen_time, social_media, sleep_hours, addiction_score]]
    prediction = model.predict(new_data)[0]
    risk_level = categorize_risk(prediction)
    
    st.success(f"Predicted Mental Health Risk Score: {prediction:.2f}")
    st.write(f"Risk Category: {risk_level}")
