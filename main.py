# main.py

from synthetictrainingdata import generate_raw_synthetic_data
from itms import engineer_features
from model_builder import train_model

# Step 1: Generate raw synthetic data
df = generate_raw_synthetic_data(n_countries=20)

# Step 2: Apply feature engineering
X_train, y_train, X_test, test_df, features = engineer_features(df)

# Step 3: Train the model
model = train_model(X_train, y_train, features)

# Step 4: Make predictions on test set
test_df['Predicted_Revenue_2024'] = model.predict(X_test)

# Step 5: Output results
print(test_df[['Country', 'Predicted_Revenue_2024']])
