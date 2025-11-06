# 02_exploratory_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load data from CSV file"""
    try:
        data = pd.read_csv('ab_test_data.csv')
        print("✅ DATA LOADED SUCCESSFULLY")
        return data
    except FileNotFoundError:
        print("❌ ERROR: ab_test_data.csv not found. Run 01_data_generation.py first.")
        return None

def perform_eda(data):
    """Perform Exploratory Data Analysis"""
    print("\n🔍 PERFORMING EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    # Check data quality
    print("=== DATA QUALITY CHECK ===")
    print(f"Missing values:\n{data.isnull().sum()}")
    print(f"Duplicate users: {data['user_id'].duplicated().sum()}")
    
    # Create visualizations
    print("\n📈 CREATING EXPLORATORY VISUALIZATIONS...")
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Click rates by group
    sns.barplot(x='group', y='clicked', data=data, ax=axes[0,0], errorbar=None, 
                hue='group', palette=['#1f77b4', '#ff7f0e'], legend=False)
    axes[0,0].set_title('Click-Through Rate by Group', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Click Rate')

    # Plot 2: Age distribution by group
    sns.histplot(data=data, x='age', hue='group', ax=axes[0,1], alpha=0.6, palette=['#1f77b4', '#ff7f0e'])
    axes[0,1].set_title('Age Distribution by Group', fontsize=14, fontweight='bold')

    # Plot 3: Country distribution
    country_group = pd.crosstab(data['country'], data['group'], normalize='index')
    country_group.plot(kind='bar', ax=axes[1,0], color=['#1f77b4', '#ff7f0e'])
    axes[1,0].set_title('Country Distribution by Group', fontsize=14, fontweight='bold')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend(title='Group')

    # Plot 4: Previous channel watchers
    prev_watch = pd.crosstab(data['previously_watched_channel'], data['group'], normalize='index')
    prev_watch.plot(kind='bar', ax=axes[1,1], color=['#1f77b4', '#ff7f0e'])
    axes[1,1].set_title('Previous Channel Watchers by Group', fontsize=14, fontweight='bold')
    axes[1,1].set_xticklabels(['Not Watched', 'Watched'], rotation=0)
    axes[1,1].legend(title='Group')

    plt.tight_layout()
    plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ VISUALIZATIONS SAVED AS 'exploratory_analysis.png'")

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        perform_eda(data)