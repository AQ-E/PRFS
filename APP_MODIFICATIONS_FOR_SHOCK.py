"""
APP.PY MODIFICATIONS FOR SHOCK MODEL LOADING
=============================================

Replace the load_multimodel_assets() function in app.py with this version.
It will automatically load the shock bundle when shock mode is ON.

Find this function around line 1502 in your app.py and REPLACE it.
"""

def load_multimodel_assets(use_shock_mode=False) -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """
    Return (bundle, meta, df_hist) or (None, None, None) if absent.
    
    Parameters:
    -----------
    use_shock_mode : bool
        If True, loads tax_models_bundle_shock.pkl
        If False, loads tax_models_bundle.pkl (baseline)
    """
    
    # Determine which bundle to load
    if use_shock_mode:
        pkl_name = "tax_models_bundle_shock.pkl"
    else:
        pkl_name = "tax_models_bundle.pkl"
    
    pkl_path = _resolve(pkl_name)
    json_path = _resolve("tax_models_meta.json")
    xlsx_path = _resolve("tax_prepared_data.xlsx")
    csv_path = _resolve("tax_prepared_data.csv")

    # If shock bundle doesn't exist, fall back to baseline with warning
    if use_shock_mode and not pkl_path:
        st.warning(f"""
        ⚠️ Shock-embedded model bundle ({pkl_name}) not found.
        
        Falling back to baseline models. Shock mode will use baseline models
        with decomposed features (experimental).
        
        To create proper shock models, run: `python retrain_shock_models.py`
        """)
        pkl_name = "tax_models_bundle.pkl"
        pkl_path = _resolve(pkl_name)

    if not pkl_path or not json_path:
        return None, None, None

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if xlsx_path:
        df = load_tax_data(decompose_shocks=use_shock_mode)
    elif csv_path:
        df = pd.read_csv(csv_path)
        if 'year_end' not in df.columns and df.index.name == 'year_end':
            df = df.reset_index()
        if use_shock_mode:
            df = decompose_inflation_shocks(df)
    else:
        return bundle, meta, None

    df = prepare_transforms(df)

    # Pre-compute ENet residuals
    for head, b in bundle["models"].items():
        if "enet" in b:
            model = b["enet"]["model"]
            feat_cols = b["enet"]["feature_cols"] if "feature_cols" in b["enet"] else b["spec"]["x"]
            y_name = b["spec"]["y"]
            train_resids = []
            valid_hist = df.dropna(subset=[y_name]).index[2:]
            for t in valid_hist:
                try:
                    row = {}
                    for c in feat_cols:
                        if c in df.columns:
                            row[c] = df.loc[t, c]
                    if len(row) == len(feat_cols):
                        X_test = pd.DataFrame([row])[feat_cols]
                        yhat_t = model.predict(X_test)[0]
                        ytrue_t = df.loc[t, y_name]
                        train_resids.append(ytrue_t - yhat_t)
                except:
                    pass
            b["enet"]["train_residuals"] = np.array(train_resids) if train_resids else np.array([])

    return bundle, meta, df


# ============================================================================
# UPDATED MAIN FLOW
# ============================================================================

"""
In the main flow section (around line 2744), CHANGE this line:

FROM:
    bundle, meta, df_hist_req = load_multimodel_assets()

TO:
    bundle, meta, df_hist_req = load_multimodel_assets(use_shock_mode=use_demand_supply_shocks)


Full context (around line 2740-2750):

    import hashlib as _hashlib
    _data_version = _hashlib.md5(pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()).hexdigest()[:12]

    # MODIFIED LINE:
    bundle, meta, df_hist_req = load_multimodel_assets(use_shock_mode=use_demand_supply_shocks)
    
    if "user_df" in st.session_state:
        df_hist = st.session_state["user_df"]
    else:
        df_hist = df_hist_req

    buoy_data = load_buoyancy()
"""


# ============================================================================
# COMPLETE INSTRUCTIONS
# ============================================================================

"""
STEPS TO INTEGRATE SHOCK MODELS INTO YOUR APP:
===============================================

1. COPY FILES TO YOUR APP DIRECTORY
   ---------------------------------
   - tax_models_bundle_shock.pkl  → Same folder as app.py
   - prfs_unified_best_shock.py   → Same folder as app.py (optional)

2. MODIFY app.py
   -------------
   
   Step A: Replace load_multimodel_assets() function
           (Around line 1502)
           Copy the modified version from above
   
   Step B: Update the function call
           (Around line 2744)
           Change:
               bundle, meta, df_hist_req = load_multimodel_assets()
           To:
               bundle, meta, df_hist_req = load_multimodel_assets(use_shock_mode=use_demand_supply_shocks)

3. TEST
   ----
   - Run: streamlit run app.py
   - Toggle shock mode OFF: Should use tax_models_bundle.pkl (baseline)
   - Toggle shock mode ON: Should use tax_models_bundle_shock.pkl (shock-trained)
   - Check that forecasts differ between modes

4. DONE!
   -----
   Your app now has proper shock-embedded forecasting with
   models trained on decomposed inflation features.
"""
