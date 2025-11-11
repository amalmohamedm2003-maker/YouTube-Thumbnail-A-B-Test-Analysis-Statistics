# 05_bayesian_analysis.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load data from CSV file"""
    try:
        data = pd.read_csv('ab_test_data.csv')
        return data
    except FileNotFoundError:
        print("❌ ERROR: ab_test_data.csv not found. Run 01_data_generation.py first.")
        return None

def perform_bayesian_analysis(data):
    """Perform Bayesian A/B testing analysis"""
    print("\n🔮 PERFORMING BAYESIAN ANALYSIS")
    print("=" * 50)

    print("=== BAYESIAN A/B TESTING ===")
    # Prior parameters (weak prior)
    alpha_prior = 2
    beta_prior = 10

    # Calculate successes and trials
    control_success = data[data['group'] == 'control']['clicked'].sum()
    control_trials = data[data['group'] == 'control']['clicked'].count()

    treatment_success = data[data['group'] == 'treatment']['clicked'].sum()
    treatment_trials = data[data['group'] == 'treatment']['clicked'].count()

    # Posterior distributions
    alpha_control_post = alpha_prior + control_success
    beta_control_post = beta_prior + (control_trials - control_success)

    alpha_treatment_post = alpha_prior + treatment_success
    beta_treatment_post = beta_prior + (treatment_trials - treatment_success)

    print(f"Control posterior: Beta({alpha_control_post}, {beta_control_post})")
    print(f"Treatment posterior: Beta({alpha_treatment_post}, {beta_treatment_post})")

    # Sample from posterior distributions
    np.random.seed(42)
    n_samples = 100000
    control_samples = np.random.beta(alpha_control_post, beta_control_post, n_samples)
    treatment_samples = np.random.beta(alpha_treatment_post, beta_treatment_post, n_samples)

    # Calculate probability that treatment is better
    prob_treatment_better = (treatment_samples > control_samples).mean()
    difference = treatment_samples - control_samples

    print(f"\n📊 BAYESIAN RESULTS:")
    print(f"Probability that treatment is better: {prob_treatment_better:.4f} ({(prob_treatment_better * 100):.2f}%)")
    print(f"Mean improvement: {difference.mean():.4f}")
    print(f"95% credible interval: [{np.percentile(difference, 2.5):.4f}, {np.percentile(difference, 97.5):.4f}]")

    # Expected loss
    loss = np.maximum(control_samples - treatment_samples, 0).mean()
    print(f"Expected loss if we choose treatment: {loss:.6f}")

    # Create Bayesian visualization
    print("\n📈 CREATING BAYESIAN VISUALIZATIONS...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Posterior distributions
    x = np.linspace(0.08, 0.20, 1000)
    control_pdf = stats.beta.pdf(x, alpha_control_post, beta_control_post)
    treatment_pdf = stats.beta.pdf(x, alpha_treatment_post, beta_treatment_post)

    ax1.plot(x, control_pdf, label='Control (Old Thumbnail)', linewidth=2, color='#1f77b4')
    ax1.plot(x, treatment_pdf, label='Treatment (New Thumbnail)', linewidth=2, color='#ff7f0e')
    ax1.fill_between(x, control_pdf, alpha=0.2, color='#1f77b4')
    ax1.fill_between(x, treatment_pdf, alpha=0.2, color='#ff7f0e')
    ax1.set_xlabel('Click-Through Rate')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Posterior Distributions of CTR', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Distribution of differences
    ax2.hist(difference, bins=50, alpha=0.7, edgecolor='black', color='#2ca02c')
    ax2.axvline(0, color='red', linestyle='--', label='No difference', linewidth=2)
    ax2.axvline(difference.mean(), color='blue', linestyle='-', label='Mean difference', linewidth=2)
    ax2.set_xlabel('Treatment CTR - Control CTR')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Treatment Effect', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bayesian_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ BAYESIAN VISUALIZATIONS SAVED AS 'bayesian_analysis.png'")
    
    return {
        'bayesian_probability': prob_treatment_better,
        'mean_improvement': difference.mean(),
        'credible_interval_lower': np.percentile(difference, 2.5),
        'credible_interval_upper': np.percentile(difference, 97.5),
        'expected_loss': loss
    }

if __name__ == "__main__":
    data = load_data()
    if data is not None:
        bayesian_results = perform_bayesian_analysis(data)