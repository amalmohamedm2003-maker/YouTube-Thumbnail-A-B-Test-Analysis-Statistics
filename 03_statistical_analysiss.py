# 03_statistical_testing.py
import pandas as pd
from scipy import stats

def load_data():
    """Load data from CSV file"""
    try:
        data = pd.read_csv('ab_test_data.csv')
        return data
    except FileNotFoundError:
        print("❌ ERROR: ab_test_data.csv not found. Run 01_data_generation.py first.")
        return None

def perform_statistical_tests(data):
    """Perform basic statistical tests"""
    print("\n📊 PERFORMING STATISTICAL TESTING")
    print("=" * 50)
    
    # Separate groups
    control_clicks = data[data['group'] == 'control']['clicked']
    treatment_clicks = data[data['group'] == 'treatment']['clicked']

    # T-test
    print("=== INDEPENDENT T-TEST RESULTS ===")
    t_stat, p_value = stats.ttest_ind(treatment_clicks, control_clicks)

    abs_diff = treatment_clicks.mean() - control_clicks.mean()
    rel_improvement = (abs_diff / control_clicks.mean()) * 100

    print(f"Control CTR:    {control_clicks.mean():.4f} ({control_clicks.sum()}/{len(control_clicks)} clicks)")
    print(f"Treatment CTR:  {treatment_clicks.mean():.4f} ({treatment_clicks.sum()}/{len(treatment_clicks)} clicks)")
    print(f"Absolute difference: {abs_diff:.4f}")
    print(f"Relative improvement: {rel_improvement:.2f}%")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("🎯 RESULT: STATISTICALLY SIGNIFICANT - New thumbnail performs better!")
    else:
        print("❌ RESULT: Not statistically significant")

    # Chi-square test
    print("\n=== CHI-SQUARE TEST VALIDATION ===")
    contingency_table = pd.crosstab(data['group'], data['clicked'])
    print("Contingency Table:")
    print(contingency_table)

    chi2, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_chi:.6f}")

    # Check covariate balance
    print("\n=== COVARIATE BALANCE CHECK ===")
    t_age, p_age = stats.ttest_ind(
        data[data['group'] == 'control']['age'],
        data[data['group'] == 'treatment']['age']
    )
    print(f"Age balance - T-stat: {t_age:.4f}, P-value: {p_age:.4f}")

    country_table = pd.crosstab(data['country'], data['group'])
    chi2_country, p_country, _, _ = stats.chi2_contingency(country_table)
    print(f"Country balance - Chi2: {chi2_country:.4f}, P-value: {p_country:.4f}")

    watch_table = pd.crosstab(data['previously_watched_channel'], data['group'])
    chi2_watch, p_watch, _, _ = stats.chi2_contingency(watch_table)
    print(f"Previous watchers - Chi2: {chi2_watch:.4f}, P-value: {p_watch:.4f}")

    # Enhanced analysis for imbalance
    print(f"\n🔍 INVESTIGATING IMBALANCE IN PREVIOUS WATCHERS:")
    watch_distribution = pd.crosstab(data['previously_watched_channel'], data['group'])
    print("Distribution of previous watchers:")
    print(watch_distribution)
    
    control_watchers_pct = (watch_distribution.loc[1, 'control'] / watch_distribution.loc[:, 'control'].sum()) * 100
    treatment_watchers_pct = (watch_distribution.loc[1, 'treatment'] / watch_distribution.loc[:, 'treatment'].sum()) * 100
    
    print(f"Control group: {control_watchers_pct:.1f}% are previous watchers")
    print(f"Treatment group: {treatment_watchers_pct:.1f}% are previous watchers")
    print(f"Difference: {abs(control_watchers_pct - treatment_watchers_pct):.1f} percentage points")

    # Balance conclusion
    print(f"\n🎯 COVARIATE BALANCE ASSESSMENT:")
    if p_age > 0.05 and p_country > 0.05 and p_watch > 0.05:
        print("✅ ALL COVARIATES ARE BALANCED - Randomization successful")
    else:
        print("⚠️  COVARIATE IMBALANCE DETECTED - Advanced analysis required")
        if p_watch <= 0.05:
            print(f"   • Previous watchers distribution differs (p={p_watch:.4f})")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Use logistic regression to control for these variables")
        print(f"   2. Check if the imbalance affects our conclusion")
    
    return {
        'control_rate': control_clicks.mean(),
        'treatment_rate': treatment_clicks.mean(),
        'absolute_difference': abs_diff,
        'relative_improvement': rel_improvement,
        'p_value': p_value,
        't_statistic': t_stat,
        'covariate_imbalanced': p_watch <= 0.05
    }

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        results = perform_statistical_tests(data)