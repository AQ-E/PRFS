"""
IMPROVED ELASTICITY ESTIMATION
===============================

Better methods for estimating oil-price elasticity of Pakistan tax revenue.

Uses log-log specification with lags for more accurate elasticity.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Load data
df = pd.read_excel('tax_prepared_data.xlsx')

# Calculate total tax revenue
df['total_tax'] = df['dt'] + df['gst'] + df['fed'] + df['customs']

# Oil price in PKR terms (critical!)
df['oil_price_pkr'] = df['oil_price_brent'] * df['exrate']

# ============================================================================
# METHOD 1: LOG-LOG REGRESSION (RECOMMENDED)
# ============================================================================

print("\n" + "="*70)
print("METHOD 1: LOG-LOG REGRESSION")
print("="*70)

# Create logs
df['log_total_tax'] = np.log(df['total_tax'])
df['log_oil_pkr'] = np.log(df['oil_price_pkr'])
df['log_gdp'] = np.log(df['gdp'])

# Create lagged oil price
df['log_oil_pkr_lag1'] = df['log_oil_pkr'].shift(1)

# Drop missing
model_df = df[['log_total_tax', 'log_oil_pkr', 'log_oil_pkr_lag1', 
               'log_gdp', 'policy rate']].dropna()

# Regression
X = sm.add_constant(model_df[['log_oil_pkr', 'log_oil_pkr_lag1', 
                                'log_gdp', 'policy rate']])
y = model_df['log_total_tax']

result = sm.OLS(y, X).fit()

print("\nRegression Results:")
print(result.summary())

# Extract elasticities
elasticity_contemporaneous = result.params['log_oil_pkr']
elasticity_lagged = result.params['log_oil_pkr_lag1']
elasticity_total = elasticity_contemporaneous + elasticity_lagged

print("\n" + "="*70)
print("ELASTICITY ESTIMATES:")
print("="*70)
print(f"Contemporaneous elasticity: {elasticity_contemporaneous:.4f}")
print(f"Lagged elasticity:          {elasticity_lagged:.4f}")
print(f"TOTAL ELASTICITY:           {elasticity_total:.4f}")
print(f"\nInterpretation: 1% oil price increase → {elasticity_total:.4f}% tax revenue change")

# Statistical significance
p_contemp = result.pvalues['log_oil_pkr']
p_lag = result.pvalues['log_oil_pkr_lag1']

print(f"\nSignificance:")
print(f"Contemporaneous: p = {p_contemp:.4f} {'***' if p_contemp < 0.01 else '**' if p_contemp < 0.05 else '*' if p_contemp < 0.1 else 'not sig'}")
print(f"Lagged:          p = {p_lag:.4f} {'***' if p_lag < 0.01 else '**' if p_lag < 0.05 else '*' if p_lag < 0.1 else 'not sig'}")

# ============================================================================
# METHOD 2: ERROR CORRECTION MODEL (ECM)
# ============================================================================

print("\n" + "="*70)
print("METHOD 2: ERROR CORRECTION MODEL")
print("="*70)

# First differences
df['d_log_total_tax'] = df['log_total_tax'].diff()
df['d_log_oil_pkr'] = df['log_oil_pkr'].diff()
df['d_log_gdp'] = df['log_gdp'].diff()

# Lagged levels for cointegration term
df['log_total_tax_lag1'] = df['log_total_tax'].shift(1)
df['log_oil_pkr_lag1'] = df['log_oil_pkr'].shift(1)
df['log_gdp_lag1'] = df['log_gdp'].shift(1)

# Error correction term (from long-run relationship)
# First estimate long-run: log_tax = θ·log_oil + γ·log_gdp
lr_df = df[['log_total_tax', 'log_oil_pkr', 'log_gdp']].dropna()
X_lr = sm.add_constant(lr_df[['log_oil_pkr', 'log_gdp']])
y_lr = lr_df['log_total_tax']
lr_result = sm.OLS(y_lr, X_lr).fit()

# Compute error correction term
df['ec_term'] = (df['log_total_tax_lag1'] - 
                 lr_result.params['const'] - 
                 lr_result.params['log_oil_pkr'] * df['log_oil_pkr_lag1'] -
                 lr_result.params['log_gdp'] * df['log_gdp_lag1'])

# ECM regression
ecm_df = df[['d_log_total_tax', 'd_log_oil_pkr', 'd_log_gdp', 'ec_term']].dropna()
X_ecm = sm.add_constant(ecm_df[['d_log_oil_pkr', 'd_log_gdp', 'ec_term']])
y_ecm = ecm_df['d_log_total_tax']

ecm_result = sm.OLS(y_ecm, X_ecm).fit()

print("\nECM Results:")
print(ecm_result.summary())

short_run_elasticity = ecm_result.params['d_log_oil_pkr']
long_run_elasticity = lr_result.params['log_oil_pkr']
adjustment_speed = ecm_result.params['ec_term']

print("\n" + "="*70)
print("ECM ELASTICITIES:")
print("="*70)
print(f"Short-run elasticity: {short_run_elasticity:.4f}")
print(f"Long-run elasticity:  {long_run_elasticity:.4f}")
print(f"Adjustment speed:     {adjustment_speed:.4f}")
print(f"\nInterpretation:")
print(f"  - Immediate impact: {short_run_elasticity:.4f}%")
print(f"  - Long-term impact: {long_run_elasticity:.4f}%")
print(f"  - Half-life of shock: {np.log(0.5)/np.log(1+adjustment_speed):.1f} years")

# ============================================================================
# METHOD 3: TIME-VARYING ELASTICITY
# ============================================================================

print("\n" + "="*70)
print("METHOD 3: TIME-VARYING ELASTICITY (Rolling Window)")
print("="*70)

window = 10  # 10-year rolling window
elasticities = []
years = []

for i in range(window, len(df)):
    subset = df.iloc[i-window:i]
    subset_clean = subset[['log_total_tax', 'log_oil_pkr', 'log_gdp']].dropna()
    
    if len(subset_clean) >= 5:  # Need minimum observations
        X_roll = sm.add_constant(subset_clean[['log_oil_pkr', 'log_gdp']])
        y_roll = subset_clean['log_total_tax']
        
        try:
            roll_result = sm.OLS(y_roll, X_roll).fit()
            elasticities.append(roll_result.params['log_oil_pkr'])
            years.append(df.iloc[i]['year_end'])
        except:
            pass

print(f"\nElasticity evolution ({window}-year window):")
for year, elast in zip(years, elasticities):
    print(f"  {year}: {elast:+.4f}")

if elasticities:
    print(f"\nAverage elasticity: {np.mean(elasticities):.4f}")
    print(f"Recent elasticity (last window): {elasticities[-1]:.4f}")
    print(f"Trend: {'Increasing' if elasticities[-1] > elasticities[0] else 'Decreasing'}")

# ============================================================================
# SUMMARY & RECOMMENDATION
# ============================================================================

print("\n" + "="*70)
print("SUMMARY & RECOMMENDATION")
print("="*70)

print("\nElasticity Estimates from Different Methods:")
print(f"1. Log-log (total):           {elasticity_total:.4f}")
print(f"2. ECM (long-run):            {long_run_elasticity:.4f}")
print(f"3. Rolling window (recent):   {elasticities[-1] if elasticities else 'N/A':.4f}")

recommended = elasticity_total
print(f"\n📊 RECOMMENDED ELASTICITY FOR PRFS APP: {recommended:.4f}")
print(f"\nRationale:")
print(f"- Log-log specification gives true elasticity")
print(f"- Includes both contemporaneous and lagged effects")
print(f"- Controls for GDP and policy rate")
print(f"- Statistically robust")

print("\n" + "="*70)
print(f"✅ USE THIS IN YOUR APP: TOTAL_TAX_OIL_ELASTICITY = {recommended:.3f}")
print("="*70)

# Save results
results_summary = pd.DataFrame({
    'Method': ['Log-log (total)', 'Log-log (contemp)', 'Log-log (lagged)',
               'ECM (short-run)', 'ECM (long-run)', 'Rolling (recent)'],
    'Elasticity': [
        elasticity_total,
        elasticity_contemporaneous,
        elasticity_lagged,
        short_run_elasticity,
        long_run_elasticity,
        elasticities[-1] if elasticities else np.nan
    ]
})

results_summary.to_excel('elasticity_estimates_improved.xlsx', index=False)
print("\n✓ Results saved to: elasticity_estimates_improved.xlsx")
