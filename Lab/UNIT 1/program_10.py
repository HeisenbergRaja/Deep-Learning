# Program 10: Weather Forecasting: Probability Distribution Estimation & Analysis
# Question: Weather forecasting: Generate random temperature samples, estimate probability distribution, analyse if observations follow expected distribution.

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 1. Generate synthetic daily temperature observations (1 year = 365 days)
np.random.seed(42)
mu_expected = 24.0      # Expected seasonal mean temperature in Celsius
sigma_expected = 5.0    # Expected standard deviation in Celsius
n_days = 365
daily_temperatures = np.random.normal(loc=mu_expected, scale=sigma_expected, size=n_days)

# 2. Parameter Estimation from observed sample data
sample_mean = np.mean(daily_temperatures)
sample_std = np.std(daily_temperatures, ddof=1)
sample_var = np.var(daily_temperatures, ddof=1)
sample_median = np.median(daily_temperatures)
sample_skew = stats.skew(daily_temperatures)
sample_kurt = stats.kurtosis(daily_temperatures)

# 3. Weather event probability estimation (Empirical vs. Theoretical)
# Freezing/Cold day: Temp < 15°C; Heatwave day: Temp > 30°C; Mild day: 20°C to 28°C
p_cold_emp = np.mean(daily_temperatures < 15.0)
p_cold_theo = stats.norm.cdf(15.0, loc=mu_expected, scale=sigma_expected)

p_hot_emp = np.mean(daily_temperatures > 30.0)
p_hot_theo = 1.0 - stats.norm.cdf(30.0, loc=mu_expected, scale=sigma_expected)

p_mild_emp = np.mean((daily_temperatures >= 20.0) & (daily_temperatures <= 28.0))
p_mild_theo = stats.norm.cdf(28.0, loc=mu_expected, scale=sigma_expected) - stats.norm.cdf(20.0, loc=mu_expected, scale=sigma_expected)

# 4. Goodness-of-Fit Analysis: Kolmogorov-Smirnov (KS) Test
# H0: The observed samples follow the theoretical Normal(24.0, 5.0) distribution.
# H1: The observed samples do not follow the theoretical distribution.
ks_statistic, p_val = stats.kstest(daily_temperatures, 'norm', args=(mu_expected, sigma_expected))

print("=" * 55)
print("WEATHER FORECASTING: PROBABILITY DISTRIBUTION ESTIMATION")
print("=" * 55)
print(f"Total Observations (Days)      : {n_days}")
print(f"Expected Theoretical Model     : Normal(mu={mu_expected:.1f} °C, sigma={sigma_expected:.1f} °C)")
print(f"Sample Estimated Mean (x_bar)  : {sample_mean:.2f} °C (Difference: {sample_mean - mu_expected:+.2f} °C)")
print(f"Sample Estimated Std Dev (s)   : {sample_std:.2f} °C (Difference: {sample_std - sigma_expected:+.2f} °C)")
print(f"Sample Estimated Variance (s^2): {sample_var:.2f}")
print(f"Sample Median                  : {sample_median:.2f} °C")
print(f"Sample Skewness                : {sample_skew:.4f} (Near 0 => Symmetric)")
print(f"Sample Kurtosis                : {sample_kurt:.4f} (Near 0 => Mesokurtic)")
print("-" * 55)
print("WEATHER EVENT PROBABILITIES (Empirical vs. Theoretical):")
print(f"  P(Temp < 15 °C) [Cold Day]   : Empirical = {p_cold_emp:.4f} | Theoretical = {p_cold_theo:.4f}")
print(f"  P(Temp > 30 °C) [Hot Day]    : Empirical = {p_hot_emp:.4f} | Theoretical = {p_hot_theo:.4f}")
print(f"  P(20 <= Temp <= 28) [Mild]   : Empirical = {p_mild_emp:.4f} | Theoretical = {p_mild_theo:.4f}")
print("-" * 55)
print("KOLMOGOROV-SMIRNOV GOODNESS-OF-FIT TEST:")
print(f"  KS Test Statistic (D)        : {ks_statistic:.4f}")
print(f"  p-value                      : {p_val:.4f}")
if p_val > 0.05:
    print(f"  Decision                     : Fail to reject H0 (p = {p_val:.4f} > 0.05).")
    print("  Conclusion                   : Temperature observations follow the expected distribution.")
else:
    print(f"  Decision                     : Reject H0 (p = {p_val:.4f} <= 0.05).")
    print("  Conclusion                   : Temperature observations deviate significantly from expected.")

# 5. Visualizations: Histogram with Fitted PDF and ECDF
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot 1: Histogram vs PDF
ax1.hist(daily_temperatures, bins=25, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Empirical Histogram')
x_vals = np.linspace(mu_expected - 4*sigma_expected, mu_expected + 4*sigma_expected, 200)
ax1.plot(x_vals, stats.norm.pdf(x_vals, mu_expected, sigma_expected), 'r-', lw=2, label=f'Expected N({mu_expected}, {sigma_expected})')
ax1.plot(x_vals, stats.norm.pdf(x_vals, sample_mean, sample_std), 'g--', lw=2, label=f'Fitted N({sample_mean:.1f}, {sample_std:.1f})')
ax1.set_title("Temperature Distribution vs. Theoretical PDF", fontweight='bold')
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("Density")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.5)

# Plot 2: Empirical CDF vs Theoretical CDF
sorted_temps = np.sort(daily_temperatures)
ecdf = np.arange(1, n_days + 1) / n_days
ax2.plot(sorted_temps, ecdf, 'b-', lw=2, label='Empirical CDF')
ax2.plot(sorted_temps, stats.norm.cdf(sorted_temps, mu_expected, sigma_expected), 'r--', lw=2, label='Theoretical CDF')
ax2.set_title("Empirical CDF vs. Theoretical CDF", fontweight='bold')
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Cumulative Probability")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("p10_weather_distribution.png", dpi=150)
# plt.show()
plt.close()
print("Visualization saved as 'p10_weather_distribution.png'.")

# Output:
# =======================================================
# WEATHER FORECASTING: PROBABILITY DISTRIBUTION ESTIMATION
# =======================================================
# Total Observations (Days)      : 365
# Expected Theoretical Model     : Normal(mu=24.0 °C, sigma=5.0 °C)
# Sample Estimated Mean (x_bar)  : 24.05 °C (Difference: +0.05 °C)
# Sample Estimated Std Dev (s)   : 4.74 °C (Difference: -0.26 °C)
# Sample Estimated Variance (s^2): 22.47
# Sample Median                  : 24.29 °C
# Sample Skewness                : 0.1587 (Near 0 => Symmetric)
# Sample Kurtosis                : 0.5913 (Near 0 => Mesokurtic)
# -------------------------------------------------------
# WEATHER EVENT PROBABILITIES (Empirical vs. Theoretical):
#   P(Temp < 15 °C) [Cold Day]   : Empirical = 0.0247 | Theoretical = 0.0359
#   P(Temp > 30 °C) [Hot Day]    : Empirical = 0.0904 | Theoretical = 0.1151
#   P(20 <= Temp <= 28) [Mild]   : Empirical = 0.6055 | Theoretical = 0.5763
# -------------------------------------------------------
# KOLMOGOROV-SMIRNOV GOODNESS-OF-FIT TEST:
#   KS Test Statistic (D)        : 0.0426
#   p-value                      : 0.5073
#   Decision                     : Fail to reject H0 (p = 0.5073 > 0.05).
#   Conclusion                   : Temperature observations follow the expected distribution.
# Visualization saved as 'p10_weather_distribution.png'.

# ======================================================================
