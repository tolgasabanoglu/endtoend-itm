# synthetic_training_data.py
import pandas as pd
import numpy as np

def generate_raw_synthetic_data(n_countries=20):
    np.random.seed(42)

    years = list(range(2010, 2025))  # 2010 to 2024 inclusive

    data = {
        'Country': [f'Country_{i}' for i in range(n_countries)]
    }

    base_gdp = np.random.randint(1000, 15000, n_countries)
    base_spending = np.random.randint(500, 5000, n_countries)

    for year in years:
        gdp_increment = np.random.randint(100, 500, n_countries)
        spending_increment = np.random.randint(50, 200, n_countries)

        data[f'GDP_{year}'] = base_gdp + gdp_increment
        data[f'Spending_{year}'] = base_spending + spending_increment

    # Only Revenue_2024 as target
    data['Revenue_2024'] = np.random.randint(5000, 20000, n_countries)

    return pd.DataFrame(data)
