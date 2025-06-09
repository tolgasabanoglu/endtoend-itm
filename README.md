# endtoend_itm

This project simulates an end-to-end machine learning pipeline using **synthetic data** to predict **2024 revenue** for a set of countries, based on their historical **GDP** and **consumer spending** trends. It demonstrates best practices in **data generation**, **feature engineering**, **model training**, and **performance evaluation** using a modular and reproducible Python script setup.

---

## 📁 Project Structure

endtoend_itm/
├── synthetic_training_data.py # Generates and prepares the dataset
├── model_builder.py # Trains Random Forest model and plots feature importance
├── prediction.py # Main entry point to run full workflow
├── requirements.txt # Python dependencies
├── README.md # Project documentation

---

## 🚀 How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt

2. Run the prediction pipeline
python prediction.py

Project Steps Explained
1. Data Generation
To simulate a realistic dataset, we created 20 countries with synthetic values:

GDP (2018–2024)

Consumer Spending (2018–2024)

Revenue for 2024 (target variable)

Each value is randomly generated but remains within plausible ranges:

GDP: $1,000–$15,000

Spending: $500–$5,000

Revenue: $5,000–$20,000 (2024 only)

Growth patterns were simulated with slight annual increments to mimic real-world economic changes.

2. Feature Engineering
To reduce noise and capture underlying trends:

📊 GDP Features (2018–2023)
Mean_GDP_Last_5Y

Std_GDP_Last_5Y

Q10_GDP_Last_5Y (10th percentile)

Q75_GDP_Last_5Y (75th percentile)

💸 Spending Features
Mean_Spending_Last_5Y

Std_Spending_Last_5Y

📈 Growth Features
GDP_Growth_Last_5Y = (GDP_2023 - GDP_2018) / GDP_2018

Spending_Growth_Last_5Y = (Spending_2023 - Spending_2018) / Spending_2018

📍 Real-Time Feature
Spending_2024 is included directly as a strong signal feature for revenue.

3. Train-Test Split
To evaluate generalization:

Train set = first 15 countries

Test set = last 5 countries
This allows predictions for unseen countries based on learned patterns.

4. Model Training
We use a Random Forest Regressor from sklearn:

Handles non-linear relationships well

Supports built-in feature importance

Evaluated using:

OOB (Out-of-Bag) score – estimates generalization

R² score – how well model fits the training data

5. Feature Importance
After training, the model shows which features were most influential in predicting revenue. This is visualized with a horizontal bar chart, helping interpret the model.

6. Predictions
The trained model then predicts Revenue_2024 for countries 15–19. For example:

Country	Predicted Revenue 2024
Country_15	15,230.5
Country_16	13,875.0
Country_17	12,545.2
Country_18	14,180.1
Country_19	15,450.8

📈 Summary
Step	Description
Data Generation	Created 20 synthetic countries with GDP, Spending, Revenue
Feature Engineering	Derived statistical & growth-based features
Model Training	Random Forest used with historical patterns
Evaluation	Measured with R² and Out-of-Bag scores
Prediction	Applied model to forecast 2024 revenue