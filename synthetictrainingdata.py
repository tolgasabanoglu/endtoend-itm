# synthetic_training_data.py
import pandas as pd
import numpy as np

def generate_synthetic_data(n_countries=20):
    np.random.seed(42)

    gdp_2018 = np.random.randint(1000, 15000, n_countries)
    gdp_2019 = gdp_2018 + np.random.randint(100, 500, n_countries)
    gdp_2020 = gdp_2018 + np.random.randint(100, 500, n_countries)
    gdp_2021 = gdp_2018 + np.random.randint(100, 500, n_countries)
    gdp_2022 = gdp_2018 + np.random.randint(100, 500, n_countries)
    gdp_2023 = gdp_2018 + np.random.randint(100, 500, n_countries)
    gdp_2024 = gdp_2018 + np.random.randint(100, 500, n_countries)

    spending_2018 = np.random.randint(500, 5000, n_countries)
    spending_2019 = spending_2018 + np.random.randint(50, 200, n_countries)
    spending_2020 = spending_2018 + np.random.randint(50, 200, n_countries)
    spending_2021 = spending_2018 + np.random.randint(50, 200, n_countries)
    spending_2022 = spending_2018 + np.random.randint(50, 200, n_countries)
    spending_2023 = spending_2018 + np.random.randint(50, 200, n_countries)
    spending_2024 = spending_2018 + np.random.randint(50, 200, n_countries)

    revenue_2024 = np.random.randint(5000, 20000, n_countries)

    data = {
        'Country': [f'Country_{i}' for i in range(n_countries)],
        'GDP_2018': gdp_2018, 'GDP_2019': gdp_2019, 'GDP_2020': gdp_2020,
        'GDP_2021': gdp_2021, 'GDP_2022': gdp_2022, 'GDP_2023': gdp_2023, 'GDP_2024': gdp_2024,
        'Spending_2018': spending_2018, 'Spending_2019': spending_2019,
        'Spending_2020': spending_2020, 'Spending_2021': spending_2021,
        'Spending_2022': spending_2022, 'Spending_2023': spending_2023,
        'Spending_2024': spending_2024,
        'Revenue_2024': revenue_2024
    }

    df = pd.DataFrame(data)

    years = ['2018', '2019', '2020', '2021', '2022', '2023']

    df['Mean_GDP_Last_5Y'] = df[[f'GDP_{y}' for y in years[-5:]]].mean(axis=1)
    df['Std_GDP_Last_5Y'] = df[[f'GDP_{y}' for y in years[-5:]]].std(axis=1)
    df['Q10_GDP_Last_5Y'] = df[[f'GDP_{y}' for y in years[-5:]]].quantile(0.10, axis=1)
    df['Q75_GDP_Last_5Y'] = df[[f'GDP_{y}' for y in years[-5:]]].quantile(0.75, axis=1)

    df['Mean_Spending_Last_5Y'] = df[[f'Spending_{y}' for y in years[-5:]]].mean(axis=1)
    df['Std_Spending_Last_5Y'] = df[[f'Spending_{y}' for y in years[-5:]]].std(axis=1)

    df['GDP_Growth_Last_5Y'] = (df['GDP_2023'] - df['GDP_2018']) / df['GDP_2018']
    df['Spending_Growth_Last_5Y'] = (df['Spending_2023'] - df['Spending_2018']) / df['Spending_2018']

    features = [
        'Mean_GDP_Last_5Y', 'Std_GDP_Last_5Y', 'Q10_GDP_Last_5Y', 'Q75_GDP_Last_5Y',
        'Mean_Spending_Last_5Y', 'Std_Spending_Last_5Y', 'GDP_Growth_Last_5Y',
        'Spending_Growth_Last_5Y', 'Spending_2024'
    ]

    train_df = df[:15]
    test_df = df[15:]
    X_train = train_df[features]
    y_train = train_df['Revenue_2024']
    X_test = test_df[features]

    return X_train, y_train, X_test, test_df, features
