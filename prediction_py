# prediction.py
from synthetictrainingdata import generate_synthetic_data
from model_builder import train_model

# Load and process data
X_train, y_train, X_test, test_df, features = generate_synthetic_data()

# Train model
model = train_model(X_train, y_train, features)

# Predict and show results
test_df['Predicted_Revenue_2024'] = model.predict(X_test)
print(test_df[['Country', 'Predicted_Revenue_2024']])
