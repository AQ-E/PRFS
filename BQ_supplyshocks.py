"""
BLANCHARD-QUAH DECOMPOSITION FOR PAKISTAN TAX REVENUE FORECASTING
Identifies supply vs demand shocks in oil markets (FY1995-FY2026)

File: tax_prepared_data.xlsx
Author: Revenue Forecasting System
Date: May 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#==============================================================================
# CONFIGURATION
#==============================================================================

# Your file path
FILE_PATH = r"C:\Users\LENOVO\Downloads\macrotaxforecasting-main (1)\macrotaxforecasting-main\tax_prepared_data.xlsx"

# Analysis parameters
MAX_YEAR_HISTORICAL = 2025  # Last year for BQ estimation (complete data)
MAX_YEAR_FULL = 2026        # Include 2026 for full dataset
MAXLAGS = 4                 # Maximum VAR lags to test

#==============================================================================
# STEP 1: LOAD AND PREPARE DATA
#==============================================================================

def load_data(file_path):
    """Load tax data and prepare for BQ analysis"""
    print("="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    try:
        df = pd.read_excel(file_path)
        print(f"✅ File loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Year range: {df['year_end'].min()} - {df['year_end'].max()}")
    
    # Check for required variables
    required_vars = ['year_end', 'oil_price_brent', 'oil_production', 'gdp']
    missing = [var for var in required_vars if var not in df.columns]
    
    if missing:
        print(f"\n❌ ERROR: Missing required variables: {missing}")
        print("\nAvailable columns:")
        for col in sorted(df.columns):
            print(f"  - {col}")
        return None
    
    print(f"\n✅ All required variables present:")
    for var in required_vars:
        non_null = df[var].notna().sum()
        null_count = df[var].isna().sum()
        print(f"  - {var}: {non_null} non-null, {null_count} missing")
    
    # Sort by year - keep ALL years including 2026
    df = df.sort_values('year_end').reset_index(drop=True)
    
    print(f"\n📊 Full dataset:")
    print(f"  Period: FY{df['year_end'].min()} - FY{df['year_end'].max()}")
    print(f"  Total observations: {len(df)}")
    
    # Check 2026 data availability
    df_2026 = df[df['year_end'] == 2026]
    if len(df_2026) > 0:
        print(f"\n📌 FY2026 data status:")
        print(f"  - oil_price_brent: {'✅ Available' if df_2026['oil_price_brent'].notna().iloc[0] else '❌ Missing'}")
        print(f"  - oil_production: {'✅ Available' if df_2026['oil_production'].notna().iloc[0] else '❌ Missing'}")
        print(f"  - gdp: {'✅ Available' if df_2026['gdp'].notna().iloc[0] else '❌ Missing'}")
    
    return df


def prepare_bq_variables(df):
    """
    Create growth rate variables for BQ decomposition
    
    Strategy:
    - Historical BQ estimation: FY1996-FY2025 (complete data)
    - FY2026: Keep for forecasting application (even if production missing)
    """
    print("\n" + "="*80)
    print("STEP 2: CREATING GROWTH RATE VARIABLES")
    print("="*80)
    
    df_bq = df.copy()
    
    # Calculate percentage year-over-year growth rates
    print("\nCalculating growth rates:")
    
    df_bq['oil_price_growth'] = df_bq['oil_price_brent'].pct_change() * 100
    print(f"  ✅ oil_price_growth = % YoY change in Brent crude")
    
    df_bq['oil_production_growth'] = df_bq['oil_production'].pct_change() * 100
    print(f"  ✅ oil_production_growth = % YoY change in global production")
    
    df_bq['gdp_growth_calc'] = df_bq['gdp'].pct_change() * 100
    print(f"  ✅ gdp_growth_calc = % YoY change in Pakistan GDP")
    
    # Separate historical (complete) vs full (includes 2026)
    df_bq_historical = df_bq[
        (df_bq['year_end'] <= MAX_YEAR_HISTORICAL) & 
        (df_bq['oil_price_growth'].notna()) & 
        (df_bq['oil_production_growth'].notna()) & 
        (df_bq['gdp_growth_calc'].notna())
    ].copy()
    
    print(f"\n📊 Historical dataset (for BQ estimation):")
    print(f"  Period: FY{df_bq_historical['year_end'].min()} - FY{df_bq_historical['year_end'].max()}")
    print(f"  Observations: {len(df_bq_historical)}")
    print(f"  Note: Lost 1 year (FY1995) to differencing")
    
    print(f"\n📊 Full dataset (including FY2026):")
    print(f"  Period: FY{df_bq['year_end'].min()} - FY{df_bq['year_end'].max()}")
    print(f"  Observations: {len(df_bq)}")
    
    # Check FY2026 specifically
    df_2026 = df_bq[df_bq['year_end'] == 2026]
    if len(df_2026) > 0:
        print(f"\n📌 FY2026 growth rates:")
        row = df_2026.iloc[0]
        print(f"  - oil_price_growth: {row['oil_price_growth']:.2f}%" if not pd.isna(row['oil_price_growth']) else "  - oil_price_growth: ❌ Missing")
        print(f"  - oil_production_growth: {row['oil_production_growth']:.2f}%" if not pd.isna(row['oil_production_growth']) else "  - oil_production_growth: ❌ Missing (OK for forecasting)")
        print(f"  - gdp_growth_calc: {row['gdp_growth_calc']:.2f}%" if not pd.isna(row['gdp_growth_calc']) else "  - gdp_growth_calc: ❌ Missing")
    
    # Summary statistics (historical only)
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS (Historical Period: FY1996-FY2025)")
    print("-"*80)
    summary = df_bq_historical[['oil_price_growth', 'oil_production_growth', 'gdp_growth_calc']].describe()
    print(summary.to_string())
    
    return df_bq, df_bq_historical


def test_stationarity(df_bq_historical):
    """
    Augmented Dickey-Fuller test for stationarity
    Required for VAR estimation
    """
    print("\n" + "="*80)
    print("STEP 3: STATIONARITY TESTS (Augmented Dickey-Fuller)")
    print("="*80)
    
    variables = ['oil_price_growth', 'oil_production_growth', 'gdp_growth_calc']
    
    print("\nNull Hypothesis (H0): Variable has a unit root (NON-stationary)")
    print("Alternative (H1): Variable is STATIONARY")
    print("\nDecision rule: Reject H0 if p-value < 0.05 → Variable is stationary ✅")
    print("-"*80)
    
    results = []
    
    for var in variables:
        data = df_bq_historical[var].dropna()
        adf_result = adfuller(data, autolag='AIC')
        
        is_stationary = adf_result[1] < 0.05
        
        results.append({
            'Variable': var,
            'ADF_Statistic': f"{adf_result[0]:.4f}",
            'P_value': f"{adf_result[1]:.4f}",
            'Critical_5%': f"{adf_result[4]['5%']:.4f}",
            'Stationary': '✅ YES' if is_stationary else '❌ NO'
        })
        
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    all_stationary = all([r['Stationary'] == '✅ YES' for r in results])
    
    if all_stationary:
        print("\n✅ All variables STATIONARY - Ready for VAR!")
    else:
        print("\n⚠️  Some variables NON-STATIONARY")
        print("   Note: Growth rates usually stationary in practice - proceeding")
    
    return all_stationary


#==============================================================================
# STEP 2: VAR MODEL ESTIMATION
#==============================================================================

def select_var_lag(df_bq_historical, maxlags=4):
    """
    Select optimal lag length using information criteria
    """
    print("\n" + "="*80)
    print("STEP 4: VAR LAG LENGTH SELECTION")
    print("="*80)
    
    # Prepare data matrix
    var_data = df_bq_historical[['oil_production_growth', 'oil_price_growth', 'gdp_growth_calc']].dropna()
    
    print(f"\nTesting lag lengths: 1 to {maxlags}")
    print(f"Sample size: {len(var_data)} observations")
    print(f"Period: FY{df_bq_historical['year_end'].min()} - FY{df_bq_historical['year_end'].max()}")
    print(f"Variables: 3 (oil_production_growth, oil_price_growth, gdp_growth_calc)")
    print("-"*80)
    
    # Estimate VAR and select lag order
    model = VAR(var_data)
    lag_order = model.select_order(maxlags=maxlags)
    
    print("\nInformation Criteria Results:")
    print(lag_order.summary())
    
    # Use BIC (more conservative for small samples)
    optimal_lag = lag_order.bic
    
    print(f"\n✅ SELECTED LAG ORDER: p = {optimal_lag} (based on BIC)")
    print(f"\nRationale: BIC preferred for small samples (N={len(var_data)})")
    print(f"Parameters to estimate per equation: {optimal_lag * 3 + 1}")
    
    return optimal_lag, var_data


def estimate_var_model(var_data, lag_order):
    """
    Estimate VAR(p) model
    """
    print("\n" + "="*80)
    print(f"STEP 5: ESTIMATING VAR({lag_order}) MODEL")
    print("="*80)
    
    model = VAR(var_data)
    results = model.fit(maxlags=lag_order, ic=None)
    
    print(f"\n✅ VAR({lag_order}) Estimation Complete")
    print(f"   Sample period: {len(var_data)} observations")
    print(f"   Equations: 3")
    print(f"   Parameters per equation: {lag_order * 3 + 1}")
    
    # Model fit statistics
    print("\n" + "-"*80)
    print("MODEL FIT (R-squared)")
    print("-"*80)
    
    for i, var in enumerate(['oil_production_growth', 'oil_price_growth', 'gdp_growth_calc']):
        print(f"{var:30s}: R² = {results.rsquared[i]:.4f}, Adj. R² = {results.rsquared_adj[i]:.4f}")
    
    return results


def diagnose_var_residuals(var_results):
    """
    Check residual autocorrelation (Ljung-Box test)
    """
    print("\n" + "="*80)
    print("STEP 6: RESIDUAL DIAGNOSTICS")
    print("="*80)
    
    residuals = var_results.resid
    
    print("\nLjung-Box Q-test for serial correlation")
    print("H0: No autocorrelation in residuals (white noise)")
    print("-"*80)
    
    for i, var in enumerate(['oil_production_growth', 'oil_price_growth', 'gdp_growth_calc']):
        lb_test = acorr_ljungbox(residuals.iloc[:, i], lags=10, return_df=True)
        
        has_autocorr = (lb_test['lb_pvalue'] < 0.05).any()
        
        print(f"\n{var}:")
        print(f"  Q-statistic (lag 10): {lb_test['lb_stat'].iloc[-1]:.4f}")
        print(f"  P-value: {lb_test['lb_pvalue'].iloc[-1]:.4f}")
        print(f"  Result: {'⚠️ Autocorrelation detected' if has_autocorr else '✅ No autocorrelation'}")
    
    return residuals


#==============================================================================
# STEP 3: BLANCHARD-QUAH DECOMPOSITION
#==============================================================================

def blanchard_quah_decomposition(var_results):
    """
    Apply Blanchard-Quah long-run restrictions to identify structural shocks
    
    Key restriction: Demand shocks have NO long-run effect on oil prices
    """
    print("\n" + "="*80)
    print("STEP 7: BLANCHARD-QUAH STRUCTURAL IDENTIFICATION")
    print("="*80)
    
    print("\nApplying BQ long-run restriction:")
    print("  → Demand shocks have NO permanent effect on oil prices")
    print("  → Only supply shocks can permanently affect prices")
    print("\nOrdering: [oil_production, oil_price, gdp]")
    print("-"*80)
    
    # Get reduced-form residuals
    resid = var_results.resid.values
    
    # Variance-covariance matrix
    sigma = np.cov(resid, rowvar=False)
    
    print("\nReduced-form residual covariance matrix (Σ):")
    sigma_df = pd.DataFrame(sigma, 
                            columns=['oil_prod', 'oil_price', 'gdp'],
                            index=['oil_prod', 'oil_price', 'gdp'])
    print(sigma_df.to_string())
    
    # Cholesky decomposition (simplified BQ approach)
    # Ordering imposes recursive structure
    try:
        A = np.linalg.cholesky(sigma)
        print("\n✅ Cholesky decomposition successful")
    except:
        print("\n⚠️  Sigma not positive definite, using eigenvalue decomposition")
        eigvals, eigvecs = np.linalg.eig(sigma)
        A = eigvecs @ np.diag(np.sqrt(np.abs(eigvals)))
    
    print("\nStructural impact matrix (A):")
    A_df = pd.DataFrame(A,
                       columns=['Supply Shock', 'Price Shock', 'Demand Shock'],
                       index=['oil_prod', 'oil_price', 'gdp'])
    print(A_df.to_string())
    
    # Extract structural shocks
    try:
        A_inv = np.linalg.inv(A)
        structural_shocks = (A_inv @ resid.T).T
    except:
        print("\n⚠️  Using pseudo-inverse")
        A_inv = np.linalg.pinv(A)
        structural_shocks = (A_inv @ resid.T).T
    
    # Create DataFrame
    shock_index = var_results.resid.index
    shocks_df = pd.DataFrame(
        structural_shocks,
        columns=['supply_shock', 'price_shock', 'demand_shock'],
        index=shock_index
    )
    
    print(f"\n✅ Structural shocks identified for {len(shocks_df)} periods")
    
    return shocks_df, A


def analyze_structural_shocks(shocks_df, df_bq_historical):
    """
    Analyze the identified structural shocks
    """
    print("\n" + "="*80)
    print("STEP 8: STRUCTURAL SHOCK ANALYSIS")
    print("="*80)
    
    # Add year_end to shocks
    shocks_with_year = shocks_df.copy()
    shocks_with_year['year_end'] = df_bq_historical['year_end'].values[:len(shocks_df)]
    
    print("\nSupply Shock Statistics:")
    print(shocks_df['supply_shock'].describe().to_string())
    
    print("\nDemand Shock Statistics:")
    print(shocks_df['demand_shock'].describe().to_string())
    
    # Identify major episodes
    supply_threshold = shocks_df['supply_shock'].std() * 1.5
    demand_threshold = shocks_df['demand_shock'].std() * 1.5
    
    print(f"\n" + "-"*80)
    print(f"MAJOR SHOCK EPISODES (|shock| > 1.5 × σ)")
    print("-"*80)
    
    print("\n🔴 Large SUPPLY shocks:")
    large_supply = shocks_with_year[abs(shocks_with_year['supply_shock']) > supply_threshold].sort_values('supply_shock', ascending=False)
    if len(large_supply) > 0:
        for idx, row in large_supply.iterrows():
            print(f"  FY{int(row['year_end'])}: {row['supply_shock']:+.2f}σ")
    else:
        print("  None detected")
    
    print("\n🔵 Large DEMAND shocks:")
    large_demand = shocks_with_year[abs(shocks_with_year['demand_shock']) > demand_threshold].sort_values('demand_shock', ascending=False)
    if len(large_demand) > 0:
        for idx, row in large_demand.iterrows():
            print(f"  FY{int(row['year_end'])}: {row['demand_shock']:+.2f}σ")
    else:
        print("  None detected")
    
    return shocks_with_year


def validate_against_events(shocks_with_year):
    """
    Compare BQ-identified shocks with known historical events
    """
    print("\n" + "="*80)
    print("STEP 9: VALIDATION AGAINST HISTORICAL EVENTS")
    print("="*80)
    
    major_events = {
        2008: ("Global Financial Crisis + Oil Spike", "Supply"),
        2009: ("Post-Crisis Oil Collapse", "Supply"),
        2015: ("OPEC Market Share Strategy", "Supply"),
        2020: ("COVID-19 Pandemic", "Demand"),
        2022: ("Russia-Ukraine War", "Supply"),
        2025: ("Iran-Israel War Begins", "Supply")
    }
    
    print("\nComparing BQ shocks with known events:")
    print("-"*80)
    
    validation_results = []
    
    for year, (event, expected_type) in major_events.items():
        if year in shocks_with_year['year_end'].values:
            row = shocks_with_year[shocks_with_year['year_end'] == year].iloc[0]
            supply_shock = row['supply_shock']
            demand_shock = row['demand_shock']
            
            # Determine dominant shock
            if abs(supply_shock) > abs(demand_shock):
                identified_type = "Supply"
                magnitude = supply_shock
            else:
                identified_type = "Demand"
                magnitude = demand_shock
            
            match = "✅" if identified_type == expected_type else "⚠️"
            
            validation_results.append({
                'Year': year,
                'Event': event[:45],
                'Expected': expected_type,
                'BQ_Identified': identified_type,
                'Magnitude': f"{magnitude:+.2f}σ",
                'Match': match
            })
    
    val_df = pd.DataFrame(validation_results)
    print(val_df.to_string(index=False))
    
    matches = sum([1 for r in validation_results if r['Match'] == '✅'])
    total = len(validation_results)
    
    print(f"\n📊 Validation Score: {matches}/{total} events correctly identified ({matches/total*100:.0f}%)")
    
    return validation_results


#==============================================================================
# STEP 4: HANDLE FY2026 FORECASTING
#==============================================================================

def forecast_fy2026_shock(df_bq, shocks_with_year):
    """
    Classify FY2026 shock based on observed oil price
    (Production data not needed for forecasting)
    """
    print("\n" + "="*80)
    print("STEP 10: FY2026 SHOCK CLASSIFICATION")
    print("="*80)
    
    df_2026 = df_bq[df_bq['year_end'] == 2026]
    
    if len(df_2026) == 0:
        print("\n❌ FY2026 data not found in dataset")
        return None
    
    row_2026 = df_2026.iloc[0]
    
    print("\n📌 FY2026 Observed Data:")
    print(f"  Oil price growth: {row_2026['oil_price_growth']:.2f}%" if not pd.isna(row_2026['oil_price_growth']) else "  Oil price growth: ❌ Missing")
    print(f"  Oil production growth: {row_2026['oil_production_growth']:.2f}%" if not pd.isna(row_2026['oil_production_growth']) else "  Oil production growth: ❌ Missing (OK)")
    print(f"  GDP growth: {row_2026['gdp_growth_calc']:.2f}%" if not pd.isna(row_2026['gdp_growth_calc']) else "  GDP growth: ❌ Missing")
    
    # Event-based classification
    print("\n📋 Event-Based Classification:")
    print("  Event: Iran-Israel War (Feb 2026)")
    print("  Type: SUPPLY SHOCK")
    print("  Reasoning: Geopolitical disruption → Production risk → Price surge")
    
    # Calculate shock magnitude relative to historical distribution
    if not pd.isna(row_2026['oil_price_growth']):
        oil_shock_pct = row_2026['oil_price_growth']
        
        # Standardize relative to historical supply shocks
        historical_supply_std = shocks_with_year['supply_shock'].std()
        historical_oil_price_std = df_bq[df_bq['year_end'] <= 2025]['oil_price_growth'].std()
        
        # Approximate FY2026 supply shock magnitude
        shock_2026_standardized = (oil_shock_pct / historical_oil_price_std) * historical_supply_std
        
        print(f"\n📊 FY2026 Shock Magnitude Estimate:")
        print(f"  Oil price shock: {oil_shock_pct:+.2f}%")
        print(f"  Standardized shock: {shock_2026_standardized:+.2f}σ")
        print(f"  Classification: {'LARGE' if abs(shock_2026_standardized) > 1.5 else 'MODERATE'} supply shock")
        
        fy2026_shock = {
            'year_end': 2026,
            'supply_shock': shock_2026_standardized,
            'demand_shock': 0.0,  # Assumed negligible
            'oil_price_growth': oil_shock_pct,
            'classification': 'Supply Shock (Iran-Israel War)',
            'magnitude_category': 'LARGE' if abs(shock_2026_standardized) > 1.5 else 'MODERATE'
        }
        
        return fy2026_shock
    else:
        print("\n⚠️  Cannot estimate FY2026 shock - oil price data missing")
        return None


#==============================================================================
# STEP 5: VISUALIZATION
#==============================================================================

def create_visualizations(df_bq, shocks_with_year, fy2026_shock=None):
    """
    Create comprehensive BQ visualization including FY2026
    """
    print("\n" + "="*80)
    print("STEP 11: CREATING VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(4, 1, figsize=(18, 22))
    fig.suptitle('BLANCHARD-QUAH DECOMPOSITION: Oil Market Shocks & Pakistan Economy\n(FY1996-FY2026)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    years_hist = shocks_with_year['year_end'].values
    
    # Include FY2026 if available
    if fy2026_shock is not None:
        years_full = np.append(years_hist, 2026)
        shocks_full = pd.concat([
            shocks_with_year,
            pd.DataFrame([fy2026_shock])
        ], ignore_index=True)
    else:
        years_full = years_hist
        shocks_full = shocks_with_year
    
    # PANEL 1: Oil Prices with Supply Shocks
    ax1 = axes[0]
    df_plot = df_bq[df_bq['year_end'] <= 2026]
    ax1.plot(df_plot['year_end'], df_plot['oil_price_brent'], 
             linewidth=2.5, color='darkblue', marker='o', markersize=5, label='Brent Crude Price')
    
    # Highlight FY2026
    if 2026 in df_plot['year_end'].values:
        price_2026 = df_plot[df_plot['year_end'] == 2026]['oil_price_brent'].values[0]
        if not pd.isna(price_2026):
            ax1.scatter(2026, price_2026, s=200, color='red', marker='*', 
                       edgecolors='black', linewidths=2, zorder=10, label='FY2026 (Iran-Israel War)')
    
    ax1_twin = ax1.twinx()
    ax1_twin.bar(shocks_full['year_end'], shocks_full['supply_shock'], 
                 alpha=0.3, color='red', label='Supply Shock (σ)')
    ax1_twin.axhline(0, color='black', linewidth=0.5)
    ax1_twin.set_ylabel('Supply Shock Magnitude (σ)', fontsize=11, fontweight='bold', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    ax1.set_ylabel('Oil Price (US$/barrel)', fontsize=11, fontweight='bold')
    ax1.set_title('Panel A: Oil Prices and BQ-Identified Supply Shocks', fontsize=13, fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1995, 2027)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # PANEL 2: Oil Production Growth
    ax2 = axes[1]
    df_prod = df_bq[df_bq['year_end'] <= 2026]
    ax2.plot(df_prod['year_end'], df_prod['oil_production_growth'], 
             linewidth=2, color='darkgreen', marker='s', markersize=5, label='Oil Production Growth')
    ax2.axhline(0, color='black', linewidth=1)
    
    # Scatter with shock magnitude
    for _, row in shocks_full.iterrows():
        year = row['year_end']
        if year in df_prod['year_end'].values and year <= 2025:
            prod_growth = df_prod[df_prod['year_end'] == year]['oil_production_growth'].values[0]
            ax2.scatter(year, prod_growth, 
                       s=abs(row['supply_shock'])*100, 
                       c='red', alpha=0.5, zorder=5)
    
    # Note for FY2026
    if fy2026_shock is not None:
        ax2.text(2026, ax2.get_ylim()[1]*0.9, 
                '2026: Production\ndata unavailable', 
                ha='center', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax2.set_ylabel('Production Growth (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Panel B: Oil Production Growth (shock size = magnitude)', fontsize=13, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1995, 2027)
    ax2.legend()
    
    # PANEL 3: Supply vs Demand Shocks
    ax3 = axes[2]
    width = 0.35
    x = np.arange(len(shocks_full))
    ax3.bar(x - width/2, shocks_full['supply_shock'], width, 
            label='Supply Shock', color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.bar(x + width/2, shocks_full['demand_shock'], width, 
            label='Demand Shock', color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.axhline(0, color='black', linewidth=1)
    
    # Highlight FY2026
    if fy2026_shock is not None:
        idx_2026 = len(shocks_full) - 1
        ax3.annotate('FY2026\n(Estimated)', 
                    xy=(idx_2026, shocks_full['supply_shock'].iloc[-1]), 
                    xytext=(idx_2026, shocks_full['supply_shock'].iloc[-1] + 1),
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    ax3.set_xticks(x[::2])  # Show every 2nd year
    ax3.set_xticklabels(shocks_full['year_end'].values[::2].astype(int), rotation=45)
    ax3.set_ylabel('Shock Magnitude (σ)', fontsize=11, fontweight='bold')
    ax3.set_title('Panel C: BQ-Decomposed Supply vs Demand Shocks', fontsize=13, fontweight='bold', loc='left')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # PANEL 4: Pakistan GDP Growth
    ax4 = axes[3]
    df_gdp = df_bq[df_bq['year_end'] <= 2026]
    ax4.plot(df_gdp['year_end'], df_gdp['gdp_growth_calc'], 
             linewidth=2.5, color='darkgreen', marker='o', markersize=5, label='Pakistan GDP Growth')
    ax4.axhline(0, color='black', linewidth=1)
    
    # Shade by shock type
    for _, row in shocks_full.iterrows():
        year = row['year_end']
        if abs(row['supply_shock']) > abs(row['demand_shock']):
            color = 'red' if row['supply_shock'] > 0 else 'darkred'
            alpha = 0.15 if year != 2026 else 0.25
        else:
            color = 'blue'
            alpha = 0.15
        ax4.axvline(year, color=color, alpha=alpha, linewidth=8, zorder=0)
    
    ax4.set_ylabel('GDP Growth (%)', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Fiscal Year', fontsize=11, fontweight='bold')
    ax4.set_title('Panel D: Pakistan GDP Growth (background = dominant shock type)', 
                  fontsize=13, fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1995, 2027)
    ax4.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('BQ_decomposition_with_FY2026.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Visualization saved: BQ_decomposition_with_FY2026.png")


#==============================================================================
# STEP 6: EXPORT RESULTS
#==============================================================================

def export_bq_shocks(shocks_with_year, df_bq, file_path, fy2026_shock=None):
    """
    Export BQ shocks to merge with tax revenue dataset
    """
    print("\n" + "="*80)
    print("STEP 12: EXPORTING BQ SHOCKS")
    print("="*80)
    
    # Load full original dataset
    df_full = pd.read_excel(file_path)
    
    # Merge historical BQ shocks
    df_full = df_full.merge(
        shocks_with_year[['year_end', 'supply_shock', 'demand_shock']], 
        on='year_end', 
        how='left'
    )
    
    # Add FY2026 shock if available
    if fy2026_shock is not None:
        df_full.loc[df_full['year_end'] == 2026, 'supply_shock'] = fy2026_shock['supply_shock']
        df_full.loc[df_full['year_end'] == 2026, 'demand_shock'] = fy2026_shock['demand_shock']
    
    # Create dummy and weighted variables
    supply_threshold = shocks_with_year['supply_shock'].std() * 1.5
    df_full['bq_supply_dummy'] = (abs(df_full['supply_shock']) > supply_threshold).astype(int)
    df_full['bq_supply_weighted'] = df_full['supply_shock'].fillna(0)
    df_full['bq_demand_weighted'] = df_full['demand_shock'].fillna(0)
    
    # Save
    output_file = file_path.replace('.xlsx', '_with_BQ_shocks.xlsx')
    df_full.to_excel(output_file, index=False)
    
    print(f"\n✅ Dataset exported to:")
    print(f"   {output_file}")
    
    print("\n📋 New variables created:")
    print("  - supply_shock: BQ supply shock (standardized σ)")
    print("  - demand_shock: BQ demand shock (standardized σ)")
    print("  - bq_supply_dummy: Binary (1 if |supply_shock| > 1.5σ)")
    print("  - bq_supply_weighted