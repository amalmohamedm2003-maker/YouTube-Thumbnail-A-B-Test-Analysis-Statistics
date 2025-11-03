# 01_data_generation.py
import pandas as pd
import numpy as np


def generate_ab_test_data():
    """Generate synthetic YouTube A/B test data"""
    print("🚀 GENERATING SYNTHETIC A/B TEST DATA")
    print("=" * 50)

    np.random.seed(42)
    n_users = 10000

    data = pd.DataFrame({
        'user_id': range(n_users),
        'group': np.random.choice(['control', 'treatment'], n_users, p=[0.5, 0.5]),
        'age': np.random.randint(18, 65, n_users),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU'], n_users, p=[0.6, 0.2, 0.15, 0.05]),
        'previously_watched_channel': np.random.choice([0, 1], n_users, p=[0.7, 0.3])
    })

    # Simulate different click rates
    data['clicked'] = 0
    control_mask = data['group'] == 'control'
    treatment_mask = data['group'] == 'treatment'

    data.loc[control_mask, 'clicked'] = np.random.choice([0, 1], sum(control_mask), p=[0.88, 0.12])
    data.loc[treatment_mask, 'clicked'] = np.random.choice([0, 1], sum(treatment_mask), p=[0.84, 0.16])

    # Save to CSV for other scripts
    data.to_csv('ab_test_data.csv', index=False)

    print("✅ DATA GENERATED AND SAVED")
    print(f"Dataset shape: {data.shape}")
    print(f"File saved: ab_test_data.csv")

    # Display basic info
    print("\n📊 BASIC DATASET INFO:")
    print(f"First 5 rows:")
    print(data.head())
    print(f"\nGroup distribution:")
    print(data['group'].value_counts())
    print(f"\nClick rates by group:")
    print(data.groupby('group')['clicked'].mean())

    # Show country distribution
    print(f"\n🌍 COUNTRY DISTRIBUTION:")
    country_counts = data['country'].value_counts()
    for country, count in country_counts.items():
        percentage = (count / len(data)) * 100
        print(f"   {country}: {count} users ({percentage:.1f}%)")

    return data


if __name__ == "__main__":
    data = generate_ab_test_data()