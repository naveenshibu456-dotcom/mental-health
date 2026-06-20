# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load your dataset
df = pd.read_csv("country_2026_risk_profile-selected-columns.csv")

# 3. Choose features (X) and target (y)
X = df[["avg_screen_time", "avg_social_media_hours", 
        "avg_sleep_hours", "avg_addiction_score"]]
y = df["avg_mental_health_risk"]

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make predictions
y_pred = model.predict(X_test)

# 7. Evaluate performance
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 8. Try predicting for a new country profile
new_data = [[5.0, 2.7, 7.5, 51.0]]  # example values
prediction = model.predict(new_data)
print("Predicted mental health risk:", prediction[0])
