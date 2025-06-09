# feature_engineering.py
import pandas as pd

def engineer_features(df):
    gdp_years = [y for y in range(2020, 2025)]  # 2020–2024
    spending_years = gdp_years  # Same years for consistency

    gdp_cols = [f'GDP_{y}' for y in gdp_years]
    spending_cols = [f'Spending_{y}' for y in spending_years]

    # Basic statistics for GDP
    df['Mean_GDP_Last_5Y'] = df[gdp_cols].mean(axis=1)
    df['Std_GDP_Last_5Y'] = df[gdp_cols].std(axis=1)
    df['Q10_GDP_Last_5Y'] = df[gdp_cols].quantile(0.10, axis=1)
    df['Q25_GDP_Last_5Y'] = df[gdp_cols].quantile(0.25, axis=1)
    df['Q50_GDP_Last_5Y'] = df[gdp_cols].quantile(0.50, axis=1)
    df['Q75_GDP_Last_5Y'] = df[gdp_cols].quantile(0.75, axis=1)
    df['Q95_GDP_Last_5Y'] = df[gdp_cols].quantile(0.95, axis=1)

    # Basic statistics for Spending
    df['Mean_Spending_Last_5Y'] = df[spending_cols].mean(axis=1)
    df['Std_Spending_Last_5Y'] = df[spending_cols].std(axis=1)
    df['Q25_Spending_Last_5Y'] = df[spending_cols].quantile(0.25, axis=1)
    df['Q50_Spending_Last_5Y'] = df[spending_cols].quantile(0.50, axis=1)
    df['Q75_Spending_Last_5Y'] = df[spending_cols].quantile(0.75, axis=1)
    df['Q95_Spending_Last_5Y'] = df[spending_cols].quantile(0.95, axis=1)

    # Growth features
    df['GDP_Growth_Last_5Y'] = (df[f'GDP_{gdp_years[-1]}'] - df[f'GDP_{gdp_years[0]}']) / df[f'GDP_{gdp_years[0]}']
    df['Spending_Growth_Last_5Y'] = (df[f'Spending_{spending_years[-1]}'] - df[f'Spending_{spending_years[0]}']) / df[f'Spending_{spending_years[0]}']

    features = [
        'Mean_GDP_Last_5Y', 'Std_GDP_Last_5Y', 'Q10_GDP_Last_5Y', 'Q25_GDP_Last_5Y',
        'Q50_GDP_Last_5Y', 'Q75_GDP_Last_5Y', 'Q95_GDP_Last_5Y',
        'Mean_Spending_Last_5Y', 'Std_Spending_Last_5Y', 'Q25_Spending_Last_5Y',
        'Q50_Spending_Last_5Y', 'Q75_Spending_Last_5Y', 'Q95_Spending_Last_5Y',
        'GDP_Growth_Last_5Y', 'Spending_Growth_Last_5Y', 'Spending_2024'
    ]

    train_df = df[:15]
    test_df = df[15:]
    X_train = train_df[features]
    y_train = train_df['Revenue_2024']
    X_test = test_df[features]

    return X_train, y_train, X_test, test_df, features
