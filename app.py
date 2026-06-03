"""
Pakistan Revenue Forecasting System (PRFS) - Unified Application
=================================================================
Complete single-file implementation combining multi-model (ARDL/ARIMAX/ENet) 
and dynamic 2-step PRFS engines under a SINGLE sidebar with ONLY 4 macro sliders.

Run:
    streamlit run prfs_unified_complete.py
# Force refresh 122
"""

from __future__ import annotations

import sys
import io
import os
import json
import pickle
import math
import base64
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, ElasticNetCV, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ardl import ARDL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, kpss
from scipy import linalg

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Pakistan Revenue Forecasting System (PRFS)",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STABLE VERSION: 1.3.0 (GDP-Driven | Consumption Purged | IndentationError Fixed) 2026-03-24
# LAST REFRESH: 2026-03-23 23:15 (GDP-Based Transition)
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --primary-600: #2563EB;
    --primary-700: #1D4ED8;
    --accent-purple: #8B5CF6;
    --accent-teal: #14B8A6;
    --accent-orange: #F59E0B;
    --accent-rose: #F43F5E;

    --gray-50: #FAFBFC;
    --gray-100: #F4F6F8;
    --gray-600: #4B5563;
    --gray-800: #1F2937;

    --text-primary: #1F2937;
    --text-secondary: #4B5563;
    --text-tertiary: #6B7280;

    --surface-primary: #FFFFFF;
    --surface-secondary: #FAFBFC;
    --surface-elevated: #FFFFFF;

    --bg-page: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    --bg-sidebar: linear-gradient(180deg, #e3ebe6 0%, #cfdcd4 100%);
    --bg-sidebar-overlay: linear-gradient(
      180deg,
      rgba(255, 255, 255, 0.6) 0%,
      rgba(255, 255, 255, 0.45) 100%
    );

    --text-primary: #1F2937;
    --text-secondary: #4B5563;
    --text-tertiary: #6B7280;
    --text-muted: #9CA3AF;

    --border-light: #F0F3F6;
    --border-medium: #E5E9ED;
    --border-strong: #D1D8DE;

    --success-bg: #ECFDF5;
    --success-text: #047857;
    --success-border: #A7F3D0;

    --warning-bg: #FEF3C7;
    --warning-text: #92400E;
    --warning-border: #FCD34D;

    --danger-bg: #FEE2E2;
    --danger-text: #991B1B;
    --danger-border: #FECACA;

    --info-bg: #EFF6FF;
    --info-text: #1E40AF;
    --info-border: #BFDBFE;

    --space-1: 0.25rem;
    --space-2: 0.5rem;
    --space-3: 0.75rem;
    --space-4: 1rem;
    --space-5: 1.25rem;
    --space-6: 1.5rem;
    --space-8: 2rem;
    --space-10: 2.5rem;
    --space-12: 3rem;

    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: 350ms cubic-bezier(0.4, 0, 0.2, 1);

    --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.06);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.10), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --shadow-sidebar: 4px 0 20px rgba(0, 0, 0, 0.08);

    --radius-md: 10px;
    --radius-lg: 14px;
    --radius-xl: 20px;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, .stApp, .main, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-page) !important;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

.main .block-container {
    padding: 0rem 2.5rem 3rem;
    max-width: 1800px;
    margin: 0 auto;
    padding-top: 1rem !important;
    position: relative;
    z-index: 10;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-medium);
    box-shadow: var(--shadow-sidebar);
    position: relative;
    overflow: hidden;
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: var(--bg-sidebar-overlay);
    z-index: 1;
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem;
    padding-bottom: 2rem;
    position: relative;
    z-index: 2;
}

/* Sidebar Branding */
.sidebar-brand {
    padding: 0 var(--space-6) var(--space-3);
    border-bottom: 2px solid rgba(255, 255, 255, 0.4);
    margin-bottom: var(--space-4);
    position: relative;
    z-index: 2;
    text-align: center;
}

.sidebar-brand::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: var(--space-6);
    right: var(--space-6);
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--primary-400), var(--accent-purple), transparent);
}

.brand-icon-large {
    width: 160px;
    height: 160px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    flex-shrink: 0;
    margin: 0 auto 1.75rem;
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}

.brand-icon-large img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
}

.brand-text-container {
    text-align: center;
    margin-top: 0.25rem;
}

.brand-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.55rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--primary-800) 0%, var(--primary-700) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: black;
    line-height: 1.3;
    letter-spacing: -0.75px;
    margin: 0;
    padding: 0;
    text-align: center;
}

/* Sidebar Section Headers */
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.6875rem !important;
    font-weight: 800 !important;
    color: var(--text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin: var(--space-5) var(--space-6) var(--space-3) !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    position: relative;
    z-index: 2;
    padding-left: 4px;
    border-left: 3px solid var(--primary-400);
}

/* Sidebar Labels & Text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    position: relative;
    z-index: 2;
}

/* Sidebar Inputs */
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stSelectbox select,
section[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(4px);
    border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-3) var(--space-4) !important;
    font-weight: 500 !important;
    transition: var(--transition-fast) !important;
    position: relative;
    z-index: 2;
}

section[data-testid="stSidebar"] .stNumberInput input:hover,
section[data-testid="stSidebar"] .stSelectbox select:hover {
    border-color: rgba(255, 255, 255, 0.7) !important;
    background: rgba(255, 255, 255, 0.92) !important;
}

section[data-testid="stSidebar"] .stNumberInput input:focus,
section[data-testid="stSidebar"] .stSelectbox select:focus {
    border-color: var(--primary-500) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    outline: none !important;
    background: white !important;
}

/* Sidebar Radio Buttons */
section[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(4px);
    border: 1.5px solid rgba(255, 255, 255, 0.5);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    margin: var(--space-1) 0;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: var(--transition-fast);
    position: relative;
    z-index: 2;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.95);
    border-color: var(--primary-300);
    color: var(--primary-700) !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {
    background: white;
    border-color: var(--primary-500);
    color: var(--primary-700) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm);
}

/* Sidebar Slider */
section[data-testid="stSidebar"] .stSlider {
    padding: var(--space-4) 0;
    position: relative;
    z-index: 2;
}

/* Sidebar Info Box */
section[data-testid="stSidebar"] .stAlert {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.5) !important;
    border-left: 3px solid var(--primary-600) !important;
    border-radius: var(--radius-md) !important;
    padding: var(--space-4) !important;
    position: relative;
    z-index: 2;
}

/* Sidebar Buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(4px);
    border: 1.5px solid rgba(255, 255, 255, 0.5);
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
    font-weight: 600;
    transition: var(--transition-fast);
    position: relative;
    z-index: 2;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: white;
    border-color: var(--primary-500);
    color: var(--primary-700);
    box-shadow: var(--shadow-sm);
}

/* Header Styling */
.executive-header {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #E0E7FF 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-xl);
    padding: 2rem 2.5rem;
    margin: -1rem 0 2rem;
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}

.executive-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    z-index: 1;
}

.header-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.875rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

.header-subtitle {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.625rem;
    background: linear-gradient(135deg, #FFFFFF 0%, rgba(255,255,255,0.9) 100%);
    color: var(--success-text);
    padding: 0.875rem 1.5rem;
    border-radius: var(--radius-lg);
    font-size: 0.9375rem;
    font-weight: 600;
    border: 1px solid rgba(4, 120, 87, 0.2);
    box-shadow: var(--shadow-md);
}

.status-indicator {
    width: 10px;
    height: 10px;
    background: var(--success-text);
    border-radius: 50%;
    animation: pulse 2s infinite;
    box-shadow: 0 0 0 4px rgba(4, 120, 87, 0.2);
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.15); }
}

/* KPI Cards */
.metrics-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
    margin-top: 0rem;
}

/* ── KPI Cards — Unified ─────────────────────────────────────────────────── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }

.kpi-card {
    flex: 1;
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
    border: 1px solid #BFDBFE;
    border-radius: 16px;
    padding: 1.4rem 1.5rem 1.2rem;
    min-height: 178px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
    transition: transform 0.22s ease, box-shadow 0.22s ease;
}

.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.07), 0 12px 32px rgba(0,0,0,0.10);
    background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
    border-color: #93C5FD;
}

/* Left accent bar */
.kpi-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 4px;
    border-radius: 16px 0 0 16px;
    background: linear-gradient(180deg, #2563EB, #60A5FA);
}

/* Watermark circle */
.kpi-card::before {
    content: '';
    position: absolute;
    top: -28px; right: -28px;
    width: 110px; height: 110px;
    border-radius: 50%;
    background: #2563EB;
    opacity: 0.05;
    pointer-events: none;
}

.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.6rem;
}

.kpi-icon-pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    border-radius: 10px;
    font-size: 1.1rem;
    flex-shrink: 0;
    background: #DBEAFE;
}

.kpi-badge {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 0.22rem 0.55rem;
    border-radius: 99px;
    background: #DBEAFE;
    color: #1D4ED8;
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    color: #2563EB;
}

.kpi-value {
    font-size: 1.95rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.15;
    letter-spacing: -0.5px;
    margin-bottom: 0.7rem;
}

.kpi-footer {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.kpi-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.22rem 0.6rem;
    border-radius: 99px;
    width: fit-content;
}
.kpi-chip.up   { background: #ECFDF5; color: #065F46; }
.kpi-chip.down { background: #FEF2F2; color: #991B1B; }
.kpi-chip.flat { background: #DBEAFE; color: #1D4ED8; }

.kpi-divider {
    height: 1px;
    background: rgba(37, 99, 235, 0.12);
    margin: 0.55rem 0 0.4rem;
}


/* Content Sections */
.content-section {
    background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}

.content-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-500), var(--accent-purple), var(--accent-teal));
    z-index: 1;
}

.content-section:hover {
    box-shadow: var(--shadow-md);
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
    padding: 1.25rem;
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.08) 0%, transparent 100%);
    margin: -2rem -2rem 1.5rem;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    border-bottom: 2px solid rgba(37, 99, 235, 0.1);
    position: relative;
    z-index: 2;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.375rem;
}

.section-subtitle {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #FFFFFF 0%, #EFF6FF 100%);
    color: var(--primary-700);
    padding: 0.75rem 1.25rem;
    border-radius: var(--radius-md);
    font-size: 0.875rem;
    font-weight: 600;
    border: 1px solid rgba(37, 99, 235, 0.2);
    box-shadow: var(--shadow-xs);
}

/* Tabs Styling (Simplified for Compatibility) */
.stTabs [role="tablist"] {
    gap: 1rem;
    padding: 0.5rem 0;
    margin-bottom: 2rem;
}

.stTabs [role="tab"] {
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: var(--radius-md) !important;
}

.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: var(--shadow-md) !important;
}

/* 
   Advanced Contextual UI Logic (CSS-Only)
   Hides 'Tax Revenue Stream' filter (3rd element in sidebar) 
   ONLY when All Categories (Tab 1) or Buoyancy (Tab 2) are selected.
   Keeps 'Forecasting Model' (4th element) visible on all tabs.
*/
body:has(div.stTabs [role="tablist"] button[aria-selected="true"]:nth-child(1)) 
    section[data-testid="stSidebar"] div.element-container:nth-child(3),
body:has(div.stTabs [role="tablist"] button[aria-selected="true"]:nth-child(2)) 
    section[data-testid="stSidebar"] div.element-container:nth-child(3) {
    display: none !important;
}

/* Ensure the space fills correctly when hidden and prevent overlap */
section[data-testid="stSidebar"] div.element-container:nth-child(3) + div.element-container {
    margin-top: 0 !important;
}

/* Table Styling */
/* ── DataFrame / Table — Professional Redesign ──────────────────────────── */
div[data-testid="stDataFrame"],
div.stDataFrame {
    background: white;
    border-radius: 14px;
    overflow: hidden;
    margin: 1.25rem 0;
    border: 1px solid #E5E9EF;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04), 0 6px 20px rgba(0,0,0,0.05);
    transition: box-shadow 0.25s ease;
}

div[data-testid="stDataFrame"]:hover,
div.stDataFrame:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.07), 0 16px 40px rgba(0,0,0,0.08);
    border-color: #C7D7F0;
}

.dataframe {
    width: 100%;
    border-collapse: collapse;
    border-spacing: 0;
    font-size: 0.875rem;
    background: white;
}

/* Header */
.dataframe thead {
    background: linear-gradient(180deg, #1E3A5F 0%, #1a3354 100%);
}

.dataframe thead tr th {
    background: transparent !important;
    color: #E8F0FE !important;
    font-weight: 700 !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.11em !important;
    padding: 0.9rem 1.1rem !important;
    text-align: left !important;
    border: none !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 10 !important;
    white-space: nowrap;
}

.dataframe thead tr th:last-child {
    border-right: none !important;
}

/* Body rows */
.dataframe tbody tr:nth-child(even) {
    background-color: #F7F9FC !important;
}

.dataframe tbody tr:nth-child(odd) {
    background-color: #FFFFFF !important;
}

.dataframe tbody tr:hover {
    background: linear-gradient(90deg, #EEF4FF 0%, #F5F8FF 100%) !important;
    box-shadow: inset 3px 0 0 #2563EB;
}

/* Cells */
.dataframe tbody td {
    padding: 0.75rem 1.1rem !important;
    font-weight: 500 !important;
    color: #1F2937 !important;
    border: none !important;
    border-bottom: 1px solid #F0F3F7 !important;
    border-right: 1px solid #F0F3F7 !important;
    text-align: left !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.5;
    transition: background 0.15s ease;
}

.dataframe tbody td:last-child {
    border-right: none !important;
}

/* First column — row label */
.dataframe tbody td:first-child {
    font-weight: 700 !important;
    color: #374151 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    background: inherit;
    border-left: none !important;
}

/* Numeric columns — right-align with monospace */
.dataframe tbody td:nth-child(2),
.dataframe tbody td:nth-child(3),
.dataframe tbody td:nth-child(4),
.dataframe tbody td:nth-child(5),
.dataframe tbody td:nth-child(6),
.dataframe tbody td:nth-child(7),
.dataframe tbody td:nth-child(8),
.dataframe tbody td:nth-child(9),
.dataframe tbody td:nth-child(10) {
    text-align: right !important;
    font-family: 'Space Grotesk', monospace !important;
    font-weight: 600 !important;
    color: #111827 !important;
    letter-spacing: -0.01em;
}

/* Last body row — no bottom border */
.dataframe tbody tr:last-child td {
    border-bottom: none !important;
}

/* Insight Panel */
.insight-panel {
    background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #E0E7FF 100%);
    border: 1px solid rgba(37, 99, 235, 0.2);
    border-left: 4px solid var(--primary-600);
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    margin: 2.5rem 0;
    box-shadow: var(--shadow-md);
    position: relative;
    overflow: hidden;
}

.insight-panel::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(37, 99, 235, 0.08) 0%, transparent 70%);
    z-index: 0;
    pointer-events: none;
}

.insight-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.25rem;
    position: relative;
    z-index: 1;
}

.insight-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--primary-600) 0%, var(--accent-purple) 100%);
    color: white;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

.insight-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.125rem;
    font-weight: 700;
    color: var(--primary-700);
}

.insight-content {
    font-size: 1rem;
    color: var(--text-primary);
    line-height: 1.8;
    position: relative;
    z-index: 1;
}

/* Category Cards (Tab 2) */
.category-card {
    background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
    position: relative;
    overflow: hidden;
}

.category-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-500), var(--accent-purple));
    z-index: 1;
}

.category-card:hover {
    box-shadow: var(--shadow-lg);
    transform: translateY(-2px);
    border-color: #93C5FD;
}

.category-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-light);
    position: relative;
    z-index: 2;
}

.category-icon {
    width: 52px;
    height: 52px;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.5rem;
    box-shadow: var(--shadow-md);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    flex-shrink: 0;
}

.category-card:hover .category-icon {
    transform: scale(1.12) rotate(3deg);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
}

.category-icon.customs { background: linear-gradient(135deg, #DBEAFE 0%, #BFDBFE 100%); color: var(--primary-700); }
.category-icon.gst { background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%); color: var(--accent-teal); }
.category-icon.fed { background: linear-gradient(135deg, #EDE9FE 0%, #DDD6FE 100%); color: var(--accent-purple); }
.category-icon.dt { background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%); color: var(--accent-orange); }

.category-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.375rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.25rem;
}

.category-metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-item {
    background: linear-gradient(135deg, #F0F5FF 0%, #E6EEFF 100%);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: var(--radius-md);
    padding: 1rem;
    text-align: center;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.metric-item::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-500), var(--accent-purple));
    transform: scaleX(0);
    transform-origin: left;
    transition: transform 0.25s ease;
}

.metric-item:hover {
    background: linear-gradient(135deg, #F0F5FF 0%, #E6EEFF 100%);
    border-color: var(--primary-300);
    transform: translateY(-2px);
    box-shadow: var(--shadow-sm);
}

.metric-item:hover::before {
    transform: scaleX(1);
}

.kpi-label {
    font-size: 0.8125rem;
    color: var(--text-secondary);
    font-weight: 700;
    margin-bottom: 0.625rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.625rem;
    font-weight: 700;
    color: var(--text-primary);
}

.model-info-badge {
    font-size: 0.72rem;
    color: #6B7280;
    margin: 0.75rem 0;
    padding: 0.5rem 0.7rem;
    background: #F3F4F6;
    border-radius: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.mae-badge {
    background: #FEF3C7;
    color: #92400E;
    padding: 0.25rem 0.6rem;
    border-radius: 4px;
    font-weight: 600;
}

/* Responsive */
@media (max-width: 992px) {
    .header-title { font-size: 1.625rem; }
    .kpi-value { font-size: 2.125rem; }
    .metrics-container { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
}

@media (max-width: 768px) {
    .executive-header { padding: 1.5rem 2rem; }
    .header-content { flex-direction: column; align-items: flex-start; gap: 1rem; }
    .section-header { flex-direction: column; align-items: flex-start; }
    .kpi-card { padding: 1.5rem; }
    .kpi-value { font-size: 1.875rem; }
}

@media (prefers-reduced-motion: reduce) {
    * {
        transition: none !important;
        animation: none !important;
    }
}

button:focus-visible,
input:focus-visible,
select:focus-visible {
    outline: 2px solid var(--primary-500) !important;
    outline-offset: 2px !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR LOGO FUNCTION
# ============================================================================

def get_logo_html(logo_path="img.jpg"):
    """Safely loads logo image and returns HTML with fallback."""
    logo_variants = [
        logo_path,
        "img.png",
        "img.jpeg",
        "static/img.jpg",
        "static/img.png",
        "logo.jpg",
        "logo.png",
    ]

    logo_data = None
    used_path = None

    for path in logo_variants:
        if os.path.exists(path):
            try:
                with open(path, "rb") as img_file:
                    logo_data = base64.b64encode(img_file.read()).decode()
                    used_path = path
                    break
            except Exception:
                continue

    if logo_data:
        return f'''
        <div class="sidebar-brand">
            <div class="brand-icon-large">
                <img src="data:image/{used_path.split(".")[-1]};base64,{logo_data}" alt="PRFS Logo">
            </div>
            <div class="brand-text-container">
                <div class="brand-name">PRFS</div>
            </div>
        </div>
        '''
    else:
        return '''
        <div class="sidebar-brand">
            <div class="brand-text-container">
                <div class="brand-name">PRFS</div>
            </div>
        </div>
        '''

# ============================================================================
# CONSTANTS AND LABELS
# ============================================================================

TAX_LABELS = {
    "customs": "Customs Duty",
    "dt": "Income / Direct Tax (DT)",
    "fed": "Federal Excise Duty (FED)",
    "gst": "Sales Tax / GST",
}

MODEL_LABELS = {
    "ardl": "ARDL",
    "arimax": "ARIMAX (SARIMAX)",
    "enet": "ElasticNet",
    "dynamic": "Dynamic Structural Model (DSM)",
}

MODEL_ICONS = {
    "ardl": "📊",
    "arimax": "📈",
    "enet": "🎯",
    "dynamic": "🏗️",
}

# Buoyancy mapping
B_MAP = {
    "dt": "dt",
    "gst": "st_total",
    "customs": "customs",
    "fed": "fed",
    "total": "total",
}

HEAD_LABELS = {
    "dt": "Direct Taxes (DT)",
    "gst": "Sales Tax (GST)",
    "fed": "Federal Excise Duty (FED)",
    "customs": "Customs Duty (CD)",
    "total": "TOTAL TAX REVENUE",
}

# DSM Variable glossary
VAR_GLOSSARY = {
    "log_imports_hat": (
        "Predicted Imports (log)",
        "Total predicted imports, estimated in Stage 1 from GDP, exchange rate, "
        "policy rate and inflation. The '_hat' suffix means this is a **model-predicted** "
        "value, not the raw observed imports. Using predicted values ensures that the "
        "tax equation captures only the **structural** relationship, free of measurement noise."
    ),
    "log_dutiable_imports_hat": (
        "Predicted Dutiable Imports (log)",
        "Predicted dutiable imports (the taxable subset of total imports), estimated in Stage 1. "
        "This is the primary tax base for customs duty — only goods subject to tariffs."
    ),
    "log_gdp_hat": (
        "Predicted GDP (log)",
        "Predicted nominal GDP from Stage 1 channel equations. Represents the overall "
        "size of the economy that drives direct tax collections."
    ),
    "log_lsm_hat": (
        "Predicted Large-Scale Manufacturing (log)",
        "Predicted LSM index from Stage 1. LSM is a proxy for industrial/manufacturing "
        "activity which drives corporate profits and hence direct/excise taxes."
    ),
    "inflation": (
        "Inflation Rate (%)",
        "Consumer price inflation. Higher inflation mechanically raises nominal tax "
        "collections even without real growth (price effect on ad-valorem taxes)."
    ),
    "log_exrate": (
        "Exchange Rate (log, PKR/USD)",
        "Log of the PKR/USD exchange rate. Depreciation raises the PKR value of imports, "
        "increasing customs duty and import-related GST collections."
    ),
    "policy rate": (
        "SBP Policy Rate (%)",
        "State Bank of Pakistan's benchmark interest rate. Affects investment, consumption, "
        "and import demand through the cost-of-borrowing channel."
    ),
}

# Head-specific structural explanations
HEAD_EXPLANATIONS = {
    "customs": (
        "Customs Duty is modelled as a function of **predicted dutiable imports** and/or "
        "**total imports** (from Stage 1), plus the exchange rate and inflation. "
        "The logic: tariff revenue depends on the value of goods crossing the border. "
        "Stage 1 first predicts imports from macro fundamentals (GDP, exchange rate), "
        "then Stage 2 links customs duty to those predicted imports."
    ),
    "dt": (
        "Direct Tax (Income Tax) is modelled as a function of **predicted GDP** and/or "
        "**LSM** (from Stage 1), plus inflation. "
        "The logic: income tax depends on the overall income level in the economy. "
        "Stage 1 predicts GDP/LSM from macro fundamentals, then Stage 2 links DT to those predictions."
    ),
    "gst": (
        "Sales Tax (GST) is modelled as a function of **GDP** and "
        "**predicted imports** (from Stage 1), plus inflation and exchange rate. "
        "The logic: GST is levied on domestic sales and imports. Stage 1 predicts "
        "macro fundamentals, then Stage 2 links GST to those predictions."
    ),
    "fed": (
        "Federal Excise Duty is modelled as a function of **predicted LSM** "
        "(from Stage 1) plus inflation. The logic: excise duties are levied on specific "
        "manufactured goods, so industrial output (LSM) is the primary driver."
    ),
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def _resolve(filename: str) -> Optional[str]:
    """Search common relative dirs for *filename*."""
    search_dirs = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        os.path.dirname(os.path.abspath(__file__)),
    ]
    for d in search_dirs:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None

def _to_year_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        years = out.index.astype(str).str.extract(r"(\d{4})")[0].astype(int)
        out.index = pd.PeriodIndex(years, freq="Y")
    except Exception:
        pass
    return out

@st.cache_data(show_spinner=False)
def load_tax_data(decompose_shocks: bool = False) -> pd.DataFrame:
    """
    Load and prepare tax revenue & macroeconomic data.
    
    Parameters:
    -----------
    decompose_shocks : bool
        If True, decompose inflation into demand, supply, and residual components
    
    Returns:
    --------
    pd.DataFrame with year_end index and all required variables
    """
    fpath = _resolve("tax_prepared_data.xlsx")
    if not fpath:
        st.error("Could not locate tax_prepared_data.xlsx")
        return pd.DataFrame()
    
    df = pd.read_excel(fpath, engine="openpyxl")
    
    # Ensure year_end column
    if 'year_end' not in df.columns:
        st.error("Missing 'year_end' column in data")
        return pd.DataFrame()
    
    df = _to_year_index(df)
    
    # Apply shock decomposition if requested
    if decompose_shocks:
        df = decompose_inflation_shocks(df)
    
    return df

    csv = _resolve("tax_prepared_data.csv")
    if csv:
        df = pd.read_csv(csv)
        # FIXED: Reset index to avoid year_end confusion
        if 'year_end' in df.columns:
            df = df.reset_index(drop=True)
        return df

    st.error("❌ Cannot find tax_prepared_data.xlsx or .csv")
    st.stop()

def get_oil_price_data():
    """
    Fetch oil price data (Brent crude annual average)
    Manual historical data (1995-2026)
    """
    oil_data = {
        'year_end': list(range(1995, 2027)),
        'oil_price': [
            17.0, 20.5, 19.1, 12.7, 17.9,  # 1995-1999
            28.5, 24.4, 25.0, 28.8, 38.3,  # 2000-2004
            54.5, 65.1, 71.1, 97.3, 61.8,  # 2005-2009
            79.5, 104.0, 105.0, 108.7, 99.0,  # 2010-2014
            52.4, 44.0, 54.8, 71.3, 64.2,  # 2015-2019
            41.8, 70.9, 100.9, 82.0, 85.0,  # 2020-2024
            90.0, 88.0  # 2025-2026 (projections)
        ]
    }
    
    df_oil = pd.DataFrame(oil_data)
    return df_oil

def prepare_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform levels; forward-fill rates and missing levels; add oil prices and supply shock indicators."""
    out = df.copy()
    
    # Add oil price data if not present
    if 'oil_price' not in out.columns:
        oil_df = get_oil_price_data()
        out = out.merge(oil_df, on='year_end', how='left')
    
    levels = ['dt', 'gst', 'fed', 'customs', 'gdp', 'imports', 'dutiable_imports', 'lsm', 'exrate', 'oil_price']
    rates = ["inflation", "policy rate"]

    for col in rates:
        actual = next((c for c in df.columns if c.lower() == col.lower()), None)
        if actual:
            out[actual] = pd.to_numeric(out[actual], errors="coerce").ffill()
            out[col] = out[actual]

    for col in levels:
        actual = next((c for c in out.columns if c.lower() == col.lower()), None)
        if actual:
            s = pd.to_numeric(out[actual], errors="coerce")
            s = s.ffill()
            s[s <= 0] = np.nan
            out[actual] = s
            out[f"log_{col}"] = np.log(s)
    
    # Add supply shock indicators
    if 'supply_shock' not in out.columns:
        out['supply_shock'] = 0.0
        # Historical supply shocks
        out.loc[out['year_end'] == 2008, 'supply_shock'] = 1.0  # Food/oil crisis
        out.loc[out['year_end'] == 2022, 'supply_shock'] = 0.8  # Floods
        # Current/projected shocks (Iran-Israel war)
        out.loc[out['year_end'] == 2024, 'supply_shock'] = 0.9
        out.loc[out['year_end'] == 2025, 'supply_shock'] = 0.6
        out.loc[out['year_end'] == 2026, 'supply_shock'] = 0.3

    out = out.replace([np.inf, -np.inf], np.nan)
    
    # FIXED: Set PeriodIndex ONLY at the very end, after all operations
    if 'year_end' in out.columns:
        out = out.sort_values('year_end').reset_index(drop=True)
        out.index = pd.PeriodIndex(out["year_end"], freq="Y")
    
    return out

@st.cache_data(show_spinner=False)
def load_buoyancy() -> Optional[Dict]:
    # Cache bust: Reload updated buoyancy_estimates.xlsx
    path = _resolve("buoyancy_estimates.xlsx")
    if not path:
        return None
    try:
        raw = pd.read_excel(path, sheet_name="Sensitivity Analysis", header=None, engine="openpyxl")
        base_idx = proj_idx = -1
        for i, row in raw.iterrows():
            txt = str(row[1]).lower() if len(row) > 1 else ""
            if "expected base" in txt:
                base_idx = i
            if "projections" in txt and "2026-27" in txt:
                proj_idx = i
        if base_idx == -1:
            base_idx = 12
        if proj_idx == -1:
            proj_idx = 14
        base_vals = raw.iloc[base_idx, 2:9].tolist()
        proj_vals = raw.iloc[proj_idx, 2:9].tolist()
        return {
            "fy2026_base": dict(
                dt=base_vals[0], st_domestic=base_vals[1], st_imports=base_vals[2],
                st_total=base_vals[3], customs=base_vals[4], fed=base_vals[5], total=base_vals[6],
            ),
            "fy2027_buoyancy": dict(
                dt=proj_vals[0], st_domestic=proj_vals[1], st_imports=proj_vals[2],
                st_total=proj_vals[3], customs=proj_vals[4], fed=proj_vals[5], total=proj_vals[6],
            ),
        }
    except Exception:
        return None

# ============================================================================
# SVAR SUPPLY SHOCK ANALYSIS (APPROACH 1)
# ============================================================================

@st.cache_data(show_spinner=False)

# ============================================================================
# DEMAND & SUPPLY SHOCK DECOMPOSITION
# ============================================================================

def decompose_inflation_shocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose headline inflation into demand-driven, supply-driven, and residual components.
    
    Methodology:
    ------------
    inflation = alpha + beta1*consumption_growth + beta2*lsm_growth + beta3*imports_growth
                + beta4*exrate_growth + beta5*oil_price_growth + beta6*policy_rate + error
    
    Then:
    - demand_inflation_component = beta1*consumption_growth + beta2*lsm_growth + beta3*imports_growth
    - supply_inflation_component = beta4*exrate_growth + beta5*oil_price_growth + beta6*policy_rate
    - inflation_residual_shock = actual_inflation - predicted_inflation
    
    Returns:
    --------
    DataFrame with added columns:
    - consumption_growth, lsm_growth, imports_growth, exrate_growth, oil_price_growth
    - demand_inflation_component, supply_inflation_component
    - predicted_inflation, inflation_residual_shock
    """
    
    df = df.copy()
    
    # Calculate growth rates (year-over-year percentage change)
    df['consumption_growth'] = df['consumption'].pct_change() * 100
    df['lsm_growth'] = df['lsm'].pct_change() * 100
    df['imports_growth'] = df['imports'].pct_change() * 100
    df['exrate_growth'] = df['exrate'].pct_change() * 100
    
    # Handle oil price - check if available
    if 'oil_price_brent' in df.columns:
        df['oil_price_growth'] = df['oil_price_brent'].pct_change() * 100
    else:
        df['oil_price_growth'] = 0.0  # Fallback if not available
    
    # Prepare regression data (drop NaN rows for estimation)
    reg_df = df[['inflation', 'consumption_growth', 'lsm_growth', 'imports_growth', 
                  'exrate_growth', 'oil_price_growth', 'policy rate']].copy()
    
    # Use data where all variables are available for regression
    reg_df_clean = reg_df.dropna()
    
    if len(reg_df_clean) < 10:
        # Not enough data - return with zero components
        df['demand_inflation_component'] = 0.0
        df['supply_inflation_component'] = 0.0
        df['predicted_inflation'] = df['inflation']
        df['inflation_residual_shock'] = 0.0
        return df
    
    # Run regression
    from sklearn.linear_model import LinearRegression
    
    X = reg_df_clean[['consumption_growth', 'lsm_growth', 'imports_growth', 
                       'exrate_growth', 'oil_price_growth', 'policy rate']]
    y = reg_df_clean['inflation']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Extract coefficients
    alpha = model.intercept_
    beta1, beta2, beta3, beta4, beta5, beta6 = model.coef_
    
    # Store coefficients in session state for display
    if 'st' in dir():
        import streamlit as st
        st.session_state['inflation_decomp_coefs'] = {
            'intercept': alpha,
            'consumption_growth': beta1,
            'lsm_growth': beta2,
            'imports_growth': beta3,
            'exrate_growth': beta4,
            'oil_price_growth': beta5,
            'policy_rate': beta6,
            'r_squared': model.score(X, y),
            'n_obs': len(reg_df_clean)
        }
    
    # Generate components for entire dataset
    df['demand_inflation_component'] = (
        beta1 * df['consumption_growth'].fillna(0) +
        beta2 * df['lsm_growth'].fillna(0) +
        beta3 * df['imports_growth'].fillna(0)
    )
    
    df['supply_inflation_component'] = (
        beta4 * df['exrate_growth'].fillna(0) +
        beta5 * df['oil_price_growth'].fillna(0) +
        beta6 * df['policy rate'].fillna(df['policy rate'].median())
    )
    
    df['predicted_inflation'] = (
        alpha +
        df['demand_inflation_component'] +
        df['supply_inflation_component']
    )
    
    df['inflation_residual_shock'] = df['inflation'] - df['predicted_inflation']
    
    # Fill any remaining NaNs with zeros
    for col in ['demand_inflation_component', 'supply_inflation_component', 
                'predicted_inflation', 'inflation_residual_shock']:
        df[col] = df[col].fillna(0)
    
    return df


def load_multimodel_assets() -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """Return (bundle, meta, df_hist) or (None, None, None) if absent."""
    # Cache bust 5: n_test=5 expanding-window backtest meta reload
    pkl_path = _resolve("tax_models_bundle.pkl")
    json_path = _resolve("tax_models_meta.json")
    xlsx_path = _resolve("tax_prepared_data.xlsx")
    csv_path = _resolve("tax_prepared_data.csv")

    if not pkl_path or not json_path:
        return None, None, None

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if xlsx_path:
        df = load_tax_data()
    elif csv_path:
        df = pd.read_csv(csv_path)
        # FIXED: Don't set index from CSV, let prepare_transforms handle it
        if 'year_end' not in df.columns and df.index.name == 'year_end':
            df = df.reset_index()
    else:
        return bundle, meta, None

    df = prepare_transforms(df)

    # Pre-compute ENet residuals
    for head, b in bundle["models"].items():
        if "enet" in b:
            model = b["enet"]["model"]
            feat_cols = b["enet"]["feature_cols"]
            y_name = b["spec"]["y"]
            train_resids = []
            valid_hist = df.dropna(subset=[y_name]).index[2:]
            for t in valid_hist:
                try:
                    row = {}
                    for c in feat_cols:
                        if c.endswith("_L0"):
                            base = c[:-3]
                            row[c] = df.loc[t, base] if base in df.columns else 0.0
                        elif "_L" in c:
                            parts = c.rsplit("_L", 1)
                            base, lag = parts[0], int(parts[1])
                            row[c] = df.shift(lag).loc[t, base] if base in df.columns else 0.0
                        else:
                            row[c] = df.loc[t, c] if c in df.columns else 0.0
                    row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
                    pred = float(model.predict(row_df)[0])
                    train_resids.append(df.loc[t, y_name] - pred)
                except Exception:
                    continue
            b["enet"]["residuals"] = train_resids if train_resids else [0.0]

    return bundle, meta, df

# ============================================================================
# MAPPING LAYER FUNCTIONS
# ============================================================================

def _project_univariate(series: pd.Series, horizon: int) -> np.ndarray:
    clean = series.dropna()
    y = clean.values
    x = np.arange(len(y)).reshape(-1, 1)
    m = LinearRegression().fit(x, y)
    return m.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1))

def _yearly(val, h: int) -> float:
    """Get the target for year index *h* (0-based)."""
    if isinstance(val, (list, tuple)):
        return val[h] if h < len(val) else val[-1]
    return float(val)

def build_multimodel_future_exog_from_dynamic(
    df_hist: pd.DataFrame,
    horizon: int,
    spec_x: List[str],
    targets_dict: Dict,
    covid_on: bool = False,
    regime_on: bool = False,
    use_univariate: bool = False,
    elasticities: Dict | None = None,
    use_shock_mode: bool = False,
) -> pd.DataFrame:
    """
    Build the exog future dataframe needed by the multi-model engine,
    using ONLY the four dynamic PRFS macro targets.
    """
    
    # SHOCK MODE MODIFICATION: Replace 'inflation' with decomposed components
    if use_shock_mode:
        new_spec_x = []
        for col in spec_x:
            if col == 'inflation':
                # Replace with decomposed components
                new_spec_x.extend(['demand_inflation_component', 
                                   'supply_inflation_component', 
                                   'inflation_residual_shock'])
            else:
                new_spec_x.append(col)
        spec_x = new_spec_x
    if elasticities is None:
        elasticities = dict(imports=1.0, lsm=1.0)

    last = df_hist.iloc[-1]
    last_year = int(df_hist.index.max().year)
    years = [last_year + i for i in range(1, horizon + 1)]
    idx = pd.PeriodIndex(years, freq="Y")
    fut = pd.DataFrame(index=idx)

    _map = {
        "log_gdp":        ("gdp_growth",    1.0),
        "log_imports":    ("imports_growth", elasticities.get("imports", 1.0)),
        "log_lsm":        ("lsm_growth",     elasticities.get("lsm",     1.0)),
        "log_exrate":     ("exrate_growth",  1.0),
    }

    for col, (driver_key, elast) in _map.items():
        if col not in df_hist.columns:
            continue
        if use_univariate:
            fut[col] = _project_univariate(df_hist[col], horizon)
        else:
            series_clean = df_hist[col].dropna()
            if len(series_clean) == 0:
                continue
            cur = float(series_clean.iloc[-1])
            vals = []
            for h in range(horizon):
                g = _yearly(targets_dict.get(driver_key, 0.0), h) * elast
                cur = cur + np.log1p(g)
                vals.append(cur)
            fut[col] = vals

    if "inflation" in spec_x or True:
        fut["inflation"] = [
            _yearly(targets_dict.get("inflation", 6.1), h)
            for h in range(horizon)
        ]

    if "policy rate" in spec_x or "policy_rate" in spec_x:
        fut["policy rate"] = [
            _yearly(targets_dict.get("policy_rate", 11.2), h)
            for h in range(horizon)
        ]
    
    # Add oil price projections
    if "log_oil_price" in spec_x:
        # Project oil prices based on current shock scenario
        base_oil = df_hist['log_oil_price'].iloc[-1] if 'log_oil_price' in df_hist.columns else 4.5
        fut["log_oil_price"] = [
            base_oil - (h * 0.02) for h in range(horizon)  # Gradual decline assumption
        ]
    
    # Add supply shock variable for 2024-2026
    if "supply_shock" in spec_x:
        shock_path = [0.9, 0.6, 0.3]  # Declining from 2024-2026
        fut["supply_shock"] = [shock_path[h] if h < len(shock_path) else 0.0 for h in range(horizon)]

    fut["covid"] = 1 if covid_on else 0
    fut["regime"] = 1 if regime_on else 0
    for d in ["step_2024", "dummy_2024", "dummy_2025"]:
        if d in df_hist.columns:
            fut[d] = 1 if d == "step_2024" else 0

    for c in spec_x:
        if c not in fut.columns:
            if c in df_hist.columns:
                clean = df_hist[c].dropna()
                fut[c] = float(clean.iloc[-1]) if len(clean) else 0.0
            elif c in last.index and pd.notna(last[c]):
                fut[c] = float(last[c])
            else:
                fut[c] = 0.0

    return fut[spec_x].copy()

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def forecast_plot(
    hist: pd.Series,
    fore: pd.DataFrame,
    title: str,
    y_label: str = "PKR Billion",
) -> go.Figure:
    """Build a standard forecast plot with historical line + forecast + CI bands.
    All values are converted from PKR Million → PKR Billion before plotting."""
    fig = go.Figure()

    # Convert million → billion
    hist_bn = hist.values / 1000.0
    fore_yhat  = fore["yhat"].values  / 1000.0
    fore_lo80  = fore["lo80"].values  / 1000.0
    fore_hi80  = fore["hi80"].values  / 1000.0
    fore_lo95  = fore["lo95"].values  / 1000.0
    fore_hi95  = fore["hi95"].values  / 1000.0

    # Force X-axis to display June 30 (Fiscal Year End) instead of Jan 1
    x_h = hist.index.map(lambda p: pd.Timestamp(year=p.year, month=6, day=30))
    x_f = fore.index.map(lambda p: pd.Timestamp(year=p.year, month=6, day=30))

    fig.add_trace(go.Scatter(
        x=x_h,
        y=hist_bn,
        mode="lines+markers",
        name="Historical",
        line=dict(color="#2563EB", width=3.5),
        marker=dict(size=8, color="#2563EB", line=dict(width=2, color="white")),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.06)',
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_f, x_f[::-1]]),
        y=np.concatenate([fore_hi95, fore_lo95[::-1]]),
        fill="toself",
        fillcolor="rgba(139, 92, 246, 0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% CI",
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_f, x_f[::-1]]),
        y=np.concatenate([fore_hi80, fore_lo80[::-1]]),
        fill="toself",
        fillcolor="rgba(139, 92, 246, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="80% CI",
    ))

    fig.add_trace(go.Scatter(
        x=x_f,
        y=fore_yhat,
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#8B5CF6", width=3.5, dash="dash"),
        marker=dict(size=9, color="#8B5CF6", line=dict(width=2, color="white")),
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="Space Grotesk, sans-serif", size=18, color="#1F2937"),
        ),
        xaxis_title="Year",
        yaxis_title=y_label,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1, zeroline=False),
    )
    return fig

def forecast_table(fore: pd.DataFrame, unit_label: str = "PKR Billion") -> pd.DataFrame:
    """Convert a forecast df (PKR Million) into a display table (PKR Billion)."""
    show = (fore / 1000.0).copy()
    show["80% interval"] = show.apply(lambda r: f"[{r.lo80:,.2f}, {r.hi80:,.2f}]", axis=1)
    show["95% interval"] = show.apply(lambda r: f"[{r.lo95:,.2f}, {r.hi95:,.2f}]", axis=1)
    return show[["yhat", "80% interval", "95% interval"]].rename(columns={"yhat": f"Forecast ({unit_label})"})

# ============================================================================
# BUOYANCY BENCHMARK FUNCTIONS
# ============================================================================

def _buoy_val(buoy: Dict, section: str, key: str) -> float:
    """Get buoyancy number, handle GST = st_domestic + st_imports."""
    if key == "gst":
        return buoy[section].get("st_domestic", 0) + buoy[section].get("st_imports", 0)
    return buoy[section].get(B_MAP.get(key, key), 0)

def render_benchmark(
    buoy_data: Dict,
    fore_head: pd.DataFrame,
    fore_total: pd.DataFrame,
    head: str,
):
    """Render buoyancy vs model benchmark inside the current Streamlit container."""
    if buoy_data is None:
        st.info("Buoyancy file not found – benchmark skipped.")
        return

    st.markdown("---")
    st.subheader("📊 FY2027 Buoyancy Benchmark")

    rows = []
    for key, fore in [(head, fore_head), ("total", fore_total)]:
        fy26 = _buoy_val(buoy_data, "fy2026_base", key)
        fy27_buoy = _buoy_val(buoy_data, "fy2027_buoyancy", key)
        fy27_model = fore["yhat"].iloc[0] / 1000.0 if len(fore) else 0
        lo95 = fore["lo95"].iloc[0] / 1000.0 if len(fore) else 0
        hi95 = fore["hi95"].iloc[0] / 1000.0 if len(fore) else 0

        rows.append({
            "Tax Head": HEAD_LABELS.get(key, key.upper()),
            "FY26 Actual (bn)": round(fy26, 1),
            "FY27 Buoyancy (bn)": round(fy27_buoy, 1),
            "FY27 Model (bn)": round(fy27_model, 1),
            "95% CI (bn)": f"{lo95:,.0f} — {hi95:,.0f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for key, fore in [(head, fore_head)]:
        fy26 = _buoy_val(buoy_data, "fy2026_base", key)
        buoy27 = _buoy_val(buoy_data, "fy2027_buoyancy", key)
        model27 = fore["yhat"].iloc[0] / 1000.0 if len(fore) else 0
        lo95 = fore["lo95"].iloc[0] / 1000.0 if len(fore) else 0
        hi95 = fore["hi95"].iloc[0] / 1000.0 if len(fore) else 0

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["FY26 Actual", "FY27 Buoyancy", "FY27 Model"],
            y=[fy26, buoy27, model27],
            marker_color=["#636e72", "#0984e3", "#e17055"],
            text=[f"{v:.1f}" for v in [fy26, buoy27, model27]],
            textposition="auto",
            error_y=dict(
                type="data", symmetric=False,
                array=[0, 0, hi95 - model27],
                arrayminus=[0, 0, model27 - lo95],
                thickness=2, width=10, color="#2d3436",
            ),
        ))
        fig.update_layout(
            title=f"{HEAD_LABELS.get(key, key)} – Buoyancy vs Model (with 95% CI)",
            yaxis_title="PKR Billion",
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# UTILITY FUNCTIONS (Diagnostics)
# ============================================================================

def _add_dual_jb(resid: pd.Series, out: Dict):
    try:
        _, p_full, _, _ = jarque_bera(resid)
        out["jb_full_p"] = float(p_full)
        if len(resid) > 1:
            _, p_trim, _, _ = jarque_bera(resid.iloc[1:])
            out["jb_trim_p"] = float(p_trim)
        else:
            out["jb_trim_p"] = None
    except Exception:
        out["jb_full_p"] = None
        out["jb_trim_p"] = None

def diagnostics_ardl(res) -> Dict:
    resid = pd.Series(res.resid).dropna()
    out: Dict = {}
    out["durbin_watson"] = float(sm.stats.stattools.durbin_watson(resid))
    try:
        lag = min(5, max(1, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        out["ljung_box_p"] = None
    _add_dual_jb(resid, out)
    try:
        exog = getattr(res.model, "exog", None)
        if exog is not None:
            bp = het_breuschpagan(resid.values, exog)
            out["breusch_pagan_p"] = float(bp[1])
        else:
            out["breusch_pagan_p"] = None
    except Exception:
        out["breusch_pagan_p"] = None
    out["n_resid"] = int(len(resid))
    return out

def diagnostics_arimax(res) -> Dict:
    resid = pd.Series(res.resid).dropna()
    out: Dict = {}
    out["aic"] = float(res.aic)
    out["bic"] = float(res.bic) if hasattr(res, "bic") else None
    out["durbin_watson"] = float(sm.stats.stattools.durbin_watson(resid))
    try:
        lag = min(5, max(1, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        out["ljung_box_p"] = None
    _add_dual_jb(resid, out)
    out["n_resid"] = int(len(resid))
    return out

def coef_table_ardl(res) -> pd.DataFrame:
    return pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "p": res.pvalues.values,
    })

def coef_table_arimax(res) -> pd.DataFrame:
    return pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "z": res.zvalues.values,
        "p": res.pvalues.values,
    })

def coef_table_enet(bundle_head: Dict) -> pd.DataFrame:
    model = bundle_head["enet"]["model"]
    feat_cols = bundle_head["enet"]["feature_cols"]
    enet = model.named_steps["enet"]
    out = pd.DataFrame({"term": feat_cols, "coef": enet.coef_})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])

# ============================================================================
# DYNAMIC ADAPTER FUNCTIONS (DSM Engine)
# ============================================================================

_pipeline_cls = None
_scenario_cls = None

def _ensure_dynamic_imports():
    global _pipeline_cls, _scenario_cls
    if _pipeline_cls is not None:
        return True
    try:
        app_root = os.path.dirname(os.path.abspath(__file__))
        if app_root not in sys.path:
            sys.path.insert(0, app_root)
        from engine.pipeline import ForecastingPipeline
        from engine.scenario import ScenarioEngine
        _pipeline_cls = ForecastingPipeline
        _scenario_cls = ScenarioEngine
        return True
    except ImportError:
        return False

def dynamic_is_available() -> bool:
    return _ensure_dynamic_imports()

def dynamic_run_pipeline(df_raw: pd.DataFrame):
    """Fit the full 2-stage pipeline and cache in session_state."""
    if not _ensure_dynamic_imports():
        st.error("Dynamic PRFS engine not found (PRFS/engine/ not on path).")
        return
    pipeline = _pipeline_cls(df_raw)
    pipeline.run_full_pipeline()
    st.session_state["dyn_pipeline"] = pipeline
    return pipeline

def dynamic_run_scenario(
    df_raw: pd.DataFrame,
    horizon: int,
    targets: Dict,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Run scenario using the cached pipeline."""
    pipeline = st.session_state.get("dyn_pipeline")
    if pipeline is None:
        st.warning("Please run the Dynamic Pipeline first.")
        st.stop()

    engine = _scenario_cls(df_raw, pipeline.best_models, pipeline.channel_models)
    results, base_df, scen_df = engine.run_scenario(horizon, targets)
    return results, base_df, scen_df

def dynamic_to_standard_df(results: Dict, head: str, horizon: int) -> pd.DataFrame:
    """
    Convert Dynamic engine results for *head* into the standardised
    forecast DataFrame with columns: yhat, lo80, hi80, lo95, hi95.
    Values are in PKR million (consistent with multi-model output).
    """
    r = results.get(head)
    if r is None:
        return pd.DataFrame()
    return pd.DataFrame({
        "yhat": r["scenario"].values * 1000,
        "lo80": r["l80"].values * 1000,
        "hi80": r["u80"].values * 1000,
        "lo95": r["l95"].values * 1000,
        "hi95": r["u95"].values * 1000,
    })

def dynamic_to_billion_df(results: Dict, head: str) -> pd.DataFrame:
    """Same but keep in Billion (for tables)."""
    r = results.get(head)
    if r is None:
        return pd.DataFrame()
    return pd.DataFrame({
        "yhat": r["scenario"].values,
        "lo80": r["l80"].values,
        "hi80": r["u80"].values,
        "lo95": r["l95"].values,
        "hi95": r["u95"].values,
    })

# ============================================================================
# BACKTEST CHART HELPER
# ============================================================================

@st.cache_data(show_spinner=False)
def _yr(idx_val) -> int:
    """Return an int year from a PeriodIndex element or a plain int/float."""
    return idx_val.year if hasattr(idx_val, "year") else int(idx_val)


def _parse_ardl_order(order):
    """
    Convert a stored ARDL order into (lags, order_dict) constructor args.

    Stored format from the bundle is typically:
        ([1], {'log_gdp': [0, 1], 'inflation': [0, 1]})
    statsmodels ARDL accepts:
        lags = int (max AR lag),  order = {col: int (max exog lag)}
    Since ARDL includes lags 0..max for each exog, max(stored_lags) reproduces it.
    """
    try:
        lags_part, exog_dict = order
        lags = max(lags_part) if isinstance(lags_part, (list, tuple)) else int(lags_part)
        od = {}
        for c, v in exog_dict.items():
            od[c] = (max(v) if isinstance(v, (list, tuple)) else int(v))
        return int(lags), od
    except Exception:
        return 1, None


def build_oos_backtest_refit(
    head: str,
    model_kind: str,
    _bundle,
    _df_hist: pd.DataFrame,
    n_test: int = 5,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    GENUINE expanding-window out-of-sample backtest for a single head/model.

    For every fold the model is RE-ESTIMATED on past data only, then used to
    forecast the next held-out year. This is a true OOS test — the model never
    sees the test year during estimation (no data leakage).

    - ardl   : statsmodels ARDL refit each fold using the bundle's stored order.
    - arimax : statsmodels SARIMAX refit each fold using the bundle's stored order,
               with enforce_stationarity=True and enforce_invertibility=True to
               prevent explosive AR / non-invertible MA solutions. Folds whose
               fitted roots sit on/inside the unit circle, or whose forecast is
               non-finite/explosive, are recorded as NaN (→ flagged Unstable).
    - enet   : ElasticNet pipeline (StandardScaler + ElasticNet) rebuilt from the
               stored feature spec and hyper-parameters, refit each fold.

    Returns (actual_series_in_PKR, oos_df[year, yhat in PKR]).
    """
    if _bundle is None or _df_hist is None:
        return pd.Series(dtype=float), pd.DataFrame()

    bundle_head = _bundle["models"][head]
    y_name = bundle_head["spec"]["y"]
    spec_x = [c for c in bundle_head["spec"]["x"] if c in _df_hist.columns]

    df = _df_hist.copy()
    total_obs = len(df)
    min_train = max(total_obs - n_test, 6)

    oos_records = []

    for split in range(min_train, total_obs):
        df_train = df.iloc[:split]
        df_test  = df.iloc[split:split + 1]
        test_year = _yr(df.index[split])

        # Require a usable target and usable exogenous values at the test row.
        # A fold skipped for MISSING DATA is not model instability — skip it
        # entirely (do not record NaN, which would flag the model unstable).
        if pd.isna(df_test[y_name].iloc[0]) or (spec_x and df_test[spec_x].isna().any().any()):
            continue

        yhat_log = np.nan
        try:
            if model_kind == "ardl":
                tr = df_train.dropna(subset=[y_name] + spec_x)
                if len(tr) < 6:
                    continue  # insufficient training data → skip fold
                lags, od = _parse_ardl_order(bundle_head["ardl"].get("order"))
                od = {c: (od.get(c, 1) if od else 1) for c in spec_x}
                res = ARDL(
                    endog=tr[y_name].reset_index(drop=True),
                    lags=lags,
                    exog=tr[spec_x].reset_index(drop=True),
                    order=od,
                    trend="c",
                ).fit()
                yhat_log = float(
                    res.forecast(steps=1, exog=df_test[spec_x].reset_index(drop=True)).iloc[0]
                )

            elif model_kind == "arimax":
                tr = df_train.dropna(subset=[y_name] + spec_x)
                if len(tr) < 6:
                    continue  # insufficient training data → skip fold
                order = bundle_head["arimax"].get("order", (1, 0, 0))
                mod = SARIMAX(
                    tr[y_name].reset_index(drop=True),
                    exog=(tr[spec_x].reset_index(drop=True) if spec_x else None),
                    order=order,
                    trend="c",
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                res = mod.fit(disp=0, maxiter=200)

                # Root check: AR roots and MA roots must lie OUTSIDE the unit
                # circle (|root| > 1). Anything on/inside => unstable fold.
                stable = True
                try:
                    if len(res.arroots) > 0 and float(np.min(np.abs(res.arroots))) <= 1.0:
                        stable = False
                    if len(res.maroots) > 0 and float(np.min(np.abs(res.maroots))) <= 1.0:
                        stable = False
                except Exception:
                    pass

                exog_f = df_test[spec_x].values if spec_x else None
                yhat_log = float(res.get_forecast(steps=1, exog=exog_f).predicted_mean.iloc[0])
                if not stable:
                    yhat_log = np.nan  # explosive/non-invertible → mark fold unstable

            elif model_kind == "enet":
                feat_cols = bundle_head["enet"]["feature_cols"]
                params    = bundle_head["enet"].get("params", {})
                parsed = []
                for c in feat_cols:
                    base, lag = c.rsplit("_L", 1)
                    parsed.append((base, int(lag)))

                # Build training matrix from past rows only (target at time t,
                # features lagged appropriately).
                rows, ys = [], []
                for t in range(1, split):
                    feat_row = []
                    ok = True
                    for base, lag in parsed:
                        j = t - lag
                        if j < 0 or base not in df.columns or pd.isna(df.iloc[j][base]):
                            ok = False; break
                        feat_row.append(float(df.iloc[j][base]))
                    yt = df.iloc[t][y_name]
                    if ok and not pd.isna(yt):
                        rows.append(feat_row); ys.append(float(yt))

                # Test feature row.
                te_feat, ok = [], True
                for base, lag in parsed:
                    j = split - lag
                    if j < 0 or base not in df.columns or pd.isna(df.iloc[j][base]):
                        ok = False; break
                    te_feat.append(float(df.iloc[j][base]))

                if len(rows) < 6 or not ok:
                    continue  # insufficient training data / missing test features → skip fold

                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("enet", ElasticNet(
                        alpha=float(params.get("alpha", 0.01)),
                        l1_ratio=float(params.get("l1_ratio", 0.2)),
                        max_iter=10000,
                    )),
                ])
                pipe.fit(np.array(rows), np.array(ys))
                yhat_log = float(pipe.predict([te_feat])[0])
            else:
                oos_records.append({"year": test_year, "yhat": np.nan}); continue

        except Exception:
            oos_records.append({"year": test_year, "yhat": np.nan}); continue

        # Stability guard on the forecast itself (backstop to the root check).
        if not np.isfinite(yhat_log) or yhat_log > _STABILITY_UPPER_LOG:
            oos_records.append({"year": test_year, "yhat": np.nan}); continue
        try:
            yhat_pkr = float(np.exp(yhat_log))
        except OverflowError:
            yhat_pkr = np.nan
        if not np.isfinite(yhat_pkr) or yhat_pkr <= _STABILITY_LOWER:
            yhat_pkr = np.nan
        oos_records.append({"year": test_year, "yhat": yhat_pkr})

    oos_df = pd.DataFrame(oos_records) if oos_records else pd.DataFrame(columns=["year", "yhat"])

    actual = np.exp(df[y_name].dropna())
    actual.index = pd.Index([_yr(p) for p in actual.index], name="year")

    return actual, oos_df




# ============================================================================
# BACKTEST METRIC HELPERS  (single source of truth for table + graph)
# ============================================================================

N_BT = 5  # Canonical expanding-window folds used by BOTH accuracy table and graph

# Instability thresholds
_STABILITY_UPPER_LOG = 40.0   # exp(40) ≈ 2.4e17 — clearly unrealistic for PKR revenue
_STABILITY_LOWER     = 0.0    # revenue must be positive
_SMAPE_EXPLOSION     = 500.0  # sMAPE > 500% → model is exploding

def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE (returns %)."""
    mask = (np.abs(actual) + np.abs(predicted)) > 0
    if mask.sum() == 0:
        return np.nan
    return float(100.0 * np.mean(
        2.0 * np.abs(actual[mask] - predicted[mask]) /
        (np.abs(actual[mask]) + np.abs(predicted[mask]))
    ))

def _rmse_pct(actual: np.ndarray, predicted: np.ndarray) -> float:
    """RMSE as % of mean actual (returns %)."""
    mean_a = np.mean(np.abs(actual))
    if mean_a == 0:
        return np.nan
    return float(100.0 * np.sqrt(np.mean((actual - predicted) ** 2)) / mean_a)


def _is_unstable(oos_df: pd.DataFrame, actual_s: pd.Series) -> bool:
    """
    Return True if the OOS predictions are unstable:
      - any NaN / Inf predicted value
      - any non-positive (negative or zero) predicted revenue
      - h1 sMAPE > _SMAPE_EXPLOSION %
    """
    if oos_df.empty or "yhat" not in oos_df.columns:
        return True
    yhats = oos_df["yhat"].values
    if not np.all(np.isfinite(yhats)):
        return True
    if np.any(yhats <= _STABILITY_LOWER):
        return True
    # Check sMAPE explosion
    common = sorted(set(actual_s.index) & set(oos_df.set_index("year").index))
    if common:
        act  = actual_s[common].values
        pred = oos_df.set_index("year").loc[common, "yhat"].values
        s = _smape(act, pred)
        if not np.isnan(s) and s > _SMAPE_EXPLOSION:
            return True
    return False


# DSM Stage-2 structural specifications (from engine/pipeline.py tax_specs).
# For a 1-step OOS backtest the realized regressors are known, so the Stage-1
# channel "hats" collapse to their actual values — these are the actual columns
# the DSM's structural equation consumes.
DSM_STAGE2_SPECS = {
    "dt":      ["log_gdp", "log_lsm", "inflation"],
    "gst":     ["log_gdp", "log_imports", "inflation", "log_exrate"],
    "fed":     ["log_lsm", "log_gdp", "inflation"],
    "customs": ["log_dutiable_imports", "log_imports", "log_exrate", "inflation"],
}


@st.cache_data(show_spinner=False)
def build_dsm_oos_backtest(
    head: str,
    _df_hist: pd.DataFrame,
    n_test: int = N_BT,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Genuine expanding-window OOS backtest for the DSM, on the SAME folds as the
    other models. Uses the DSM engine's own structural estimator
    (engine.models.ARDLModel) fit on the DSM Stage-2 specification, re-estimated
    on past data only each fold and forecasting the next held-out year with the
    realized regressors. This is the same estimator the live DSM uses for its
    Stage-2 tax equations, so the row in the ranking reflects the actual engine.

    If the engine is unavailable, falls back to a structural OLS on the same spec
    so the DSM still appears on identical folds rather than silently vanishing.

    Returns (actual_series_pkr, oos_df[year, yhat in PKR]).
    """
    if _df_hist is None or len(_df_hist) < 8:
        return pd.Series(dtype=float), pd.DataFrame()

    df = _df_hist.copy()
    y_col = f"log_{head}"
    if y_col not in df.columns:
        return pd.Series(dtype=float), pd.DataFrame()

    spec_x = [c for c in DSM_STAGE2_SPECS.get(head, []) if c in df.columns]
    if not spec_x:
        return pd.Series(dtype=float), pd.DataFrame()

    # Try to use the real DSM engine estimator.
    _ARDLModel = None
    try:
        from engine.models import ARDLModel as _ARDLModel  # noqa
    except Exception:
        _ARDLModel = None

    total_obs = len(df)
    min_train = max(total_obs - n_test, 6)
    oos_records = []

    for split in range(min_train, total_obs):
        df_train = df.iloc[:split].dropna(subset=[y_col] + spec_x)
        df_test  = df.iloc[split:split + 1]
        test_year = _yr(df.index[split])

        if len(df_train) < 6 or pd.isna(df_test[y_col].iloc[0]) or df_test[spec_x].isna().any().any():
            continue  # missing data / insufficient training → skip fold (not instability)

        yhat_log = np.nan
        try:
            if _ARDLModel is not None:
                m = _ARDLModel()
                m.fit(df_train.reset_index(drop=True), y_col, spec_x)
                yhat_log = float(
                    m.forecast(df_test[spec_x].reset_index(drop=True), steps=1).iloc[0]
                )
            else:
                # Fallback: structural OLS on the same Stage-2 spec.
                X_train = sm.add_constant(df_train[spec_x].values, has_constant="add")
                ols_res = sm.OLS(df_train[y_col].values, X_train).fit()
                X_test  = np.column_stack([[1.0], df_test[spec_x].values])
                yhat_log = float(ols_res.predict(X_test)[0])
        except Exception:
            oos_records.append({"year": test_year, "yhat": np.nan})
            continue

        if not np.isfinite(yhat_log) or yhat_log > _STABILITY_UPPER_LOG:
            oos_records.append({"year": test_year, "yhat": np.nan})
            continue
        try:
            yhat_pkr = float(np.exp(yhat_log))
        except OverflowError:
            yhat_pkr = np.nan
        if not np.isfinite(yhat_pkr) or yhat_pkr <= _STABILITY_LOWER:
            yhat_pkr = np.nan
        oos_records.append({"year": test_year, "yhat": yhat_pkr})

    oos_df = pd.DataFrame(oos_records) if oos_records else pd.DataFrame(columns=["year", "yhat"])

    actual = np.exp(df[y_col].dropna())
    actual.index = pd.Index([_yr(p) for p in actual.index], name="year")

    return actual, oos_df


@st.cache_data(show_spinner=False)
def build_all_model_backtests(
    head: str,
    _bundle,
    _df_hist: pd.DataFrame,
    n_test: int = N_BT,
) -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
    """
    Run a genuine expanding-window OOS backtest for ALL FOUR models:
      ardl, arimax, enet, dsm

    Every model is re-estimated on past data only and forecasts the next
    held-out year, on the SAME n_test folds and the SAME test years.
    This is the SINGLE source of truth for the Accuracy Table, model ranking,
    and the Actual vs Predicted graph.

    Returns: {model_name: (actual_series_pkr, oos_df)}
      - actual_series_pkr: full history in PKR
      - oos_df: columns [year, yhat] in PKR for the n_test OOS folds

    Models producing NaN/Inf/negative/explosive predictions are still
    returned (caller uses _is_unstable() to flag and exclude them).
    """
    results: Dict[str, Tuple[pd.Series, pd.DataFrame]] = {}
    if _bundle is None or _df_hist is None:
        return results

    # ── Individual models (ARDL, ARIMAX, ElasticNet) — refit per fold ─────────
    for model_kind in ("ardl", "arimax", "enet"):
        try:
            actual, oos_df = build_oos_backtest_refit(
                head, model_kind, _bundle, _df_hist, n_test=n_test
            )
            results[model_kind] = (actual, oos_df)
        except Exception:
            pass

    # ── DSM — expanding-window re-estimation via the real engine estimator ────
    try:
        dsm_actual, dsm_oos = build_dsm_oos_backtest(head, _df_hist, n_test=n_test)
        if not dsm_actual.empty:
            results["dsm"] = (dsm_actual, dsm_oos)
    except Exception:
        pass

    return results


def compute_metrics_from_backtest(
    actual_s: pd.Series,
    oos_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute OOS accuracy metrics purely from the plotted backtest data.
    These are the authoritative metrics used in the Accuracy Table.

    - smape_all : symmetric MAPE averaged over ALL OOS folds — PRIMARY ranking metric
    - h3_smape  : symmetric MAPE over the last 3 OOS folds — supporting
    - rmse_pct  : RMSE as % of mean actual over all folds — tiebreaker / supporting
    - h1_smape  : symmetric MAPE on the single most recent fold — shown for reference only
    """
    if oos_df.empty or len(actual_s) == 0:
        return {"smape_all": np.nan, "h1_smape": np.nan, "h3_smape": np.nan, "rmse_pct": np.nan, "n": 0}

    merged = oos_df.copy().set_index("year")
    common = sorted(set(actual_s.index) & set(merged.index))
    if not common:
        return {"smape_all": np.nan, "h1_smape": np.nan, "h3_smape": np.nan, "rmse_pct": np.nan, "n": 0}

    act  = actual_s[common].values
    pred = merged.loc[common, "yhat"].values

    smape_all = _smape(act, pred)
    h1_smape  = _smape(act[-1:], pred[-1:]) if len(act) >= 1 else np.nan
    h3_smape  = _smape(act[-3:], pred[-3:]) if len(act) >= 3 else np.nan
    rmse      = _rmse_pct(act, pred)

    return {"smape_all": smape_all, "h1_smape": h1_smape, "h3_smape": h3_smape,
            "rmse_pct": rmse, "n": len(common)}


def best_model_from_backtest_results(
    bt_results: Dict[str, Tuple[pd.Series, pd.DataFrame]]
) -> str:
    """
    Select the best STABLE model from live OOS backtest results.
    All four models (ardl, arimax, enet, dsm) are evaluated on the same folds.
    Models flagged as unstable are excluded from ranking.
    Falls back to 'ardl' if no valid results are available.

    Ranking: average OOS sMAPE across all folds (primary), RMSE% (tiebreaker).
    Averaging over all folds is far more reliable on this small sample than the
    single-fold h1 metric, which is dominated by one year's noise.
    """
    best_model = "ardl"
    best_key = (np.inf, np.inf)
    for model_kind, (actual_s, oos_df) in bt_results.items():
        if _is_unstable(oos_df, actual_s):
            continue  # skip unstable models
        m = compute_metrics_from_backtest(actual_s, oos_df)
        primary = m.get("smape_all", np.nan)
        tiebreak = m.get("rmse_pct", np.nan)
        if np.isnan(primary):
            continue
        key = (primary, tiebreak if not np.isnan(tiebreak) else np.inf)
        if key < best_key:
            best_key = key
            best_model = model_kind
    return best_model


# ============================================================================
# MULTI-MODEL ADAPTER FUNCTIONS
# ============================================================================

def mm_perf_table(meta: Dict) -> pd.DataFrame:
    return pd.DataFrame(meta["performance"])

def mm_best_model_by_mape(perf: pd.DataFrame, head: str) -> str:
    """Select best model strictly by out-of-sample h1 sMAPE (lowest wins)."""
    sub = perf[(perf["tax_head"] == head) & (perf["n_test"] > 0)].copy()
    if "h1_smape" in sub.columns:
        sub = sub.dropna(subset=["h1_smape"]).sort_values("h1_smape")
    else:
        sub = sub.sort_values("mae_pct")
    return str(sub.iloc[0]["model"]) if len(sub) else "ardl"

@st.cache_data(show_spinner=False)
def mm_get_cached_forecast(
    model_kind: str,
    head: str,
    horizon: int,
    exog_future_json: str,
    n_sims: int = 500,
    data_version: str = "",
    _bundle=None,
    _df_hist=None,
):
    """Run a single-head forecast using the multi-model bundle."""
    bundle_head = _bundle["models"][head]
    exog_future = pd.read_json(io.StringIO(exog_future_json)).sort_index()
    exog_future.index = pd.PeriodIndex(exog_future.index, freq="Y")

    y_name = bundle_head["spec"]["y"]
    spec_x = bundle_head["spec"]["x"]

    # ── ARDL — manual recursive forecast seeded from df_hist ──────────────
    if model_kind == "ardl":
        res    = bundle_head["ardl"]["res"]
        params = dict(res.params)

        # Parse param names into AR lags {lag: coef} and exog lags {col: {lag: coef}}
        ar_lags   = {}   # {1: 0.72, 2: 0.10, ...}
        exog_lags = {}   # {'log_gdp': {0: 0.89, 1: -0.12}, ...}
        const     = params.get("const", params.get("intercept", 0.0))

        for k, v in params.items():
            if k in ("const", "intercept"):
                continue
            if k.startswith(y_name + ".L"):
                lag = int(k[len(y_name) + 2:])
                ar_lags[lag] = v
            elif ".L" in k:
                col, lag_str = k.rsplit(".L", 1)
                lag = int(lag_str)
                exog_lags.setdefault(col, {})[lag] = v
            else:
                # No lag suffix → treat as contemporaneous exog (lag 0)
                exog_lags.setdefault(k, {})[0] = v

        max_ar_lag = max(ar_lags.keys()) if ar_lags else 1

        # ── AR Stability Enforcement ─────────────────────────────────────────
        # If sum of AR coefficients ≥ 1, the model is explosive (unit-root or
        # above). Scale them down to 0.95 to ensure mean-reversion at forecast
        # time. This is standard practice for ARDL models with near-unit-root
        # or slightly-explosive OLS estimates.
        ar_sum = sum(ar_lags.values())
        if ar_sum >= 0.99:
            scale = 0.95 / ar_sum
            ar_lags = {k: v * scale for k, v in ar_lags.items()}

        # Seed y history from df_hist (user-edited values)
        if _df_hist is not None and y_name in _df_hist.columns:
            y_hist = list(_df_hist[y_name].dropna().values)
        else:
            y_hist = list(res.fittedvalues.dropna().values)

        # Historical exog series (needed for any lag-1 exog terms at step 1)
        hist_exog = {}
        if _df_hist is not None:
            for col in exog_lags:
                if col in _df_hist.columns:
                    hist_exog[col] = list(_df_hist[col].dropna().values)

        # Recursive multi-step forecast
        yhat_log = []
        future_y = list(y_hist)   # grows with each forecasted step

        for h in range(horizon):
            exog_row = exog_future.iloc[h] if h < len(exog_future) else exog_future.iloc[-1]
            y_pred   = const

            # AR contribution — uses actual df_hist values for seeds
            for lag, coef in ar_lags.items():
                idx = -(lag)
                if abs(idx) <= len(future_y):
                    y_pred += coef * future_y[idx]

            # Exog contribution
            for col, lag_coefs in exog_lags.items():
                for lag, coef in lag_coefs.items():
                    if lag == 0:
                        # Contemporaneous — use future exog column
                        val = float(exog_row[col]) if col in exog_row.index else 0.0
                        y_pred += coef * val
                    else:
                        # Historical exog lag — go back into df_hist then exog_future
                        future_idx = h - lag   # index into exog_future (may be negative)
                        if future_idx >= 0:
                            val = float(exog_future.iloc[future_idx][col]) if col in exog_future.columns else 0.0
                        else:
                            # Still in history
                            hist_vals = hist_exog.get(col, [])
                            hist_idx  = future_idx   # negative → from end
                            val = float(hist_vals[hist_idx]) if hist_vals and abs(hist_idx) <= len(hist_vals) else 0.0
                        y_pred += coef * val

            yhat_log.append(y_pred)
            future_y.append(y_pred)

        yhat_log  = np.array(yhat_log)
        # Use parametric error variance to prevent empirical artifacts from exploding CIs
        sigma2    = float(np.var(res.resid.dropna().values[2:]))
        std_err   = float(np.sqrt(max(sigma2, 1e-12)))
        ar_coefs  = [v for _, v in sorted(ar_lags.items())]

        sims = []
        for _ in range(n_sims):
            noise = np.random.normal(0, std_err, size=horizon)
            path  = np.zeros(horizon)
            for i in range(horizon):
                ar = sum(c * path[i - (p + 1)] for p, c in enumerate(ar_coefs) if i - (p + 1) >= 0)
                path[i] = ar + noise[i]
            sims.append(np.exp(yhat_log + path))

        sims = np.array(sims)
        return pd.DataFrame(
            {
                "yhat": np.exp(yhat_log),
                "lo80": np.quantile(sims, 0.10, axis=0),
                "hi80": np.quantile(sims, 0.90, axis=0),
                "lo95": np.quantile(sims, 0.025, axis=0),
                "hi95": np.quantile(sims, 0.975, axis=0),
            },
            index=exog_future.index,
        )

    # ── ARIMAX — REFIT on full sample with stability enforcement, then forecast
    # Per the OOS design, the final forecast uses the selected model RE-ESTIMATED
    # on the full historical sample with enforce_stationarity / enforce_invertibility
    # ON, so the forward path cannot explode (this is the same estimation used in
    # the backtest, keeping backtest and forecast consistent).
    if model_kind == "arimax":
        spec_x = [c for c in bundle_head["spec"]["x"] if c in _df_hist.columns]
        order  = bundle_head["arimax"].get("order", (1, 0, 0))
        y_full = _df_hist[y_name]

        res = None
        try:
            tr = _df_hist.dropna(subset=[y_name] + spec_x)
            mod = SARIMAX(
                tr[y_name].reset_index(drop=True),
                exog=(tr[spec_x].reset_index(drop=True) if spec_x else None),
                order=order, trend="c",
                enforce_stationarity=True, enforce_invertibility=True,
            )
            res = mod.fit(disp=0, maxiter=200)
            sigma2 = float(res.params.get("sigma2", np.var(res.resid)))
        except Exception:
            res = bundle_head["arimax"]["res"]  # fallback to stored fit
            sigma2 = float(res.params.get("sigma2", np.var(res.resid.dropna())))

        std_err = float(np.sqrt(max(sigma2, 1e-12)))

        # Future exog aligned to the SAME spec_x used at fit time.
        if spec_x:
            exog_f = exog_future.reindex(columns=spec_x).fillna(0.0).values
        else:
            exog_f = None

        try:
            native   = res.get_forecast(steps=horizon, exog=exog_f)
            yhat_log = np.asarray(native.predicted_mean, dtype=float)
        except Exception:
            last_fit = float(np.asarray(res.fittedvalues, dtype=float)[-1])
            yhat_log = np.full(horizon, last_fit)

        # Forward stability guard: if any step is non-finite or explosive, flat-line
        # from the last sane value rather than emitting an overflow.
        last_log = float(np.log(np.exp(y_full.dropna()).iloc[-1]))
        for i in range(len(yhat_log)):
            if not np.isfinite(yhat_log[i]) or yhat_log[i] > _STABILITY_UPPER_LOG:
                yhat_log[i] = yhat_log[i - 1] if i > 0 else last_log

        sims = []
        for _ in range(n_sims):
            noise = np.random.normal(0, std_err, size=horizon)
            sims.append(np.exp(yhat_log + noise))
        sims = np.array(sims)

        return pd.DataFrame(
            {
                "yhat": np.exp(yhat_log),
                "lo80": np.quantile(sims, 0.10, axis=0),
                "hi80": np.quantile(sims, 0.90, axis=0),
                "lo95": np.quantile(sims, 0.025, axis=0),
                "hi95": np.quantile(sims, 0.975, axis=0),
            },
            index=exog_future.index,
        )

    # ElasticNet
    if model_kind == "enet":
        model = bundle_head["enet"]["model"]
        feat_cols = bundle_head["enet"]["feature_cols"]
        train_resids = bundle_head["enet"]["residuals"]

        work = pd.concat([_df_hist, exog_future], axis=0).ffill()
        preds_log = []
        for t in exog_future.index:
            row = {}
            for c in feat_cols:
                if c.endswith("_L0"):
                    row[c] = work.loc[t, c[:-3]] if c[:-3] in work.columns else 0.0
                elif "_L" in c:
                    parts = c.rsplit("_L", 1)
                    base = parts[0]
                    lag = int(parts[1])
                    row[c] = work.shift(lag).loc[t, base] if base in work.columns else 0.0
                else:
                    row[c] = work.loc[t, c] if c in work.columns else 0.0
            row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
            yhat_l = float(model.predict(row_df)[0])
            preds_log.append(yhat_l)
            work.loc[t, y_name] = yhat_l

        sim_paths = []
        for _ in range(n_sims):
            path = []
            work_sim = _df_hist.copy()
            for i, t in enumerate(exog_future.index):
                ex = exog_future.iloc[i: i + 1]
                row = {}
                for c in feat_cols:
                    if c.endswith("_L0"):
                        base = c[:-3]
                        row[c] = ex[base].iloc[0] if base in ex.columns else (work_sim[base].iloc[-1] if base in work_sim.columns else 0.0)
                    elif "_L" in c:
                        parts = c.rsplit("_L", 1)
                        base, lag = parts[0], int(parts[1])
                        try:
                            row[c] = work_sim[base].iloc[-lag] if base in work_sim.columns else 0.0
                        except (IndexError, KeyError):
                            row[c] = 0.0
                    else:
                        row[c] = work_sim[c].iloc[-1] if c in work_sim.columns else 0.0
                row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
                noisy = float(model.predict(row_df)[0]) + np.random.choice(train_resids)
                path.append(math.exp(noisy))
                new_row = pd.Series(index=_df_hist.columns, dtype=float)
                for c2 in exog_future.columns:
                    new_row[c2] = ex[c2].iloc[0]
                new_row[y_name] = noisy
                work_sim = pd.concat([work_sim, new_row.to_frame().T], ignore_index=True)
            sim_paths.append(path)

        sim_paths = np.array(sim_paths)
        return pd.DataFrame(
            {
                "yhat": [math.exp(p) for p in preds_log],
                "lo80": np.quantile(sim_paths, 0.10, axis=0),
                "hi80": np.quantile(sim_paths, 0.90, axis=0),
                "lo95": np.quantile(sim_paths, 0.025, axis=0),
                "hi95": np.quantile(sim_paths, 0.975, axis=0),
            },
            index=exog_future.index,
        )

    st.error(f"Unknown model kind: {model_kind}")
    st.stop()

def mm_forecast_head(
    bundle, meta, df_hist,
    head: str, chosen: str, horizon: int, n_sims: int,
    targets: Dict, elasticities: Dict,
    covid_on: bool, regime_on: bool,
    data_version: str = "",
    use_shock_mode: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build exog from dynamic targets and run the forecast."""
    spec_x = bundle["models"][head]["spec"]["x"]
    use_univariate = (head == "fed")
    exog = build_multimodel_future_exog_from_dynamic(
        df_hist, horizon, spec_x, targets,
        covid_on=covid_on, regime_on=regime_on,
        use_univariate=use_univariate,
        elasticities=elasticities,
        use_shock_mode=use_shock_mode,
    )
    return mm_get_cached_forecast(
        chosen, head, horizon, exog.to_json(), n_sims, data_version,
        _bundle=bundle, _df_hist=df_hist,
    ), exog

def mm_forecast_total(
    bundle, meta, df_hist,
    chosen_model: str, horizon: int, n_sims: int,
    targets: Dict, elasticities: Dict,
    covid_on: bool, regime_on: bool,
    data_version: str = "",
    use_shock_mode: bool = False,
) -> pd.DataFrame:
    """Sum forecasts for all sub-heads using the specified model."""
    total = None
    for h in ["customs", "dt", "fed", "gst"]:
        fore, _ = mm_forecast_head(
            bundle, meta, df_hist, h, chosen_model, horizon, n_sims,
            targets, elasticities, covid_on, regime_on, data_version,
            use_shock_mode=use_shock_mode
        )
        total = fore if total is None else total + fore
    return total

# ============================================================================
# SIDEBAR RENDERING
# ============================================================================

def render_sidebar(
    perf: pd.DataFrame | None = None,
    dynamic_available: bool = True,
    multimodel_available: bool = True,
    df_raw_ref: pd.DataFrame | None = None,
) -> Dict:
    """Draw the sidebar and return a dict of all user choices."""
    st.sidebar.markdown(get_logo_html(), unsafe_allow_html=True)

    sb = st.sidebar

    # Core Configuration
    sb.markdown("### 🎯 Core Configuration")

    head = sb.selectbox(
        "Tax Revenue Stream",
        options=list(TAX_LABELS.keys()),
        format_func=lambda k: TAX_LABELS[k],
        help="Select the tax category to analyze and forecast",
    )

    model_options = []
    if multimodel_available:
        model_options.extend(["ardl", "arimax", "enet"])
    if dynamic_available:
        model_options.append("dynamic")

    if not model_options:
        st.error("No forecasting engine available.")
        st.stop()

    model_choice = sb.selectbox(
        "Forecasting Model",
        options=model_options,
        format_func=lambda m: f"{MODEL_ICONS.get(m, '📊')} {MODEL_LABELS.get(m, m.upper())}",
        help="Choose the econometric model for forecasting",
    )

    sb.markdown("---")

    # Forecast Parameters
    sb.markdown("### 🔮 Forecast Parameters")

    col1, col2 = sb.columns([2, 1])
    with col1:
        horizon = sb.slider(
            "Horizon (Years)",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of years to forecast",
        )
    with col2:
        sb.metric("Years", horizon, border=False)

    n_sims = sb.select_slider(
        "Uncertainty Simulations",
        options=[100, 250, 500, 1000],
        value=500,
        help="Higher values = better confidence intervals (slower computation)",
    )

    sb.markdown("---")
    
    # ══════════════════════════════════════════════════════════════════════
    # Removed: Shock decomposition mode (not needed for sensitivity analysis)
    use_demand_supply_shocks = False  # Always False in clean version
    
    sb.markdown("---")

    # Macro Scenario
    sb.markdown("### 📈 Macro Scenario")

    input_mode = sb.radio(
        "Input Mode",
        ["Manual Sliders", "Macro Path Table"],
        horizontal=True,
    )

    if input_mode == "Manual Sliders":

        gdp_pct     = sb.number_input("Nominal GDP Growth Target (%)", -2.0,  20.0, 10.8, 0.1)
        inflation   = sb.number_input("Inflation Target (%)",            0.0,  40.0,  6.1, 0.1)
        exrate_pct  = sb.number_input("FX Depreciation (%)",            -5.0,  30.0,  1.0, 0.1)
        policy_rate = sb.number_input("Policy Rate Target (%)",          5.0,  30.0, 11.2, 0.1)

        gdp_growth    = gdp_pct    / 100
        exrate_growth = exrate_pct / 100

        targets = dict(
            gdp_growth=gdp_growth,
            inflation=inflation,
            exrate_growth=exrate_growth,
            policy_rate=policy_rate,
        )
    else:
        default = pd.DataFrame({
            "Year": range(1, horizon + 1),
            "GDP Growth (%)": [10.8] * horizon,
            "Inflation (%)": [6.1] * horizon,
            "FX Depreciation (%)": [1.0] * horizon,
            "Policy Rate (%)": [11.2] * horizon,
        })
        path = sb.data_editor(default, num_rows="fixed", hide_index=True)
        targets = dict(
            gdp_growth=(path["GDP Growth (%)"] / 100.0).tolist(),
            inflation=path["Inflation (%)"].tolist(),
            exrate_growth=(path["FX Depreciation (%)"] / 100.0).tolist(),
            policy_rate=path["Policy Rate (%)"].tolist(),
        )

    sb.markdown("---")

    # Advanced Settings — use sb consistently inside expander
    with sb.expander("⚙️ Advanced Settings", expanded=False):
        sb.caption(
            "These elasticities control how GDP growth maps to "
            "sub-series for the multi-model engine. Default = 1.0 (proportional)."
        )
        imports_e = sb.number_input("Imports ↔ GDP elasticity", 0.0, 3.0, 1.0, 0.1)
        lsm_e = sb.number_input("LSM ↔ GDP elasticity", 0.0, 3.0, 1.0, 0.1)

    elasticities = dict(imports=imports_e, lsm=lsm_e)

    # Policy Factors
    sb.markdown("---")
    sb.markdown("### 🎚️ Policy Factors")
    covid_on = sb.checkbox("COVID dummy = 1 in forecast", value=False)
    regime_on = sb.checkbox("Regime dummy = 1 in forecast", value=True)

    # Dataset Information
    sb.markdown("---")
    sb.markdown("### 📊 Dataset Information")
    if df_raw_ref is not None:
        max_year = df_raw_ref.index.max().year if hasattr(df_raw_ref.index, "year") else "Unknown"
        sb.info(
            f"**Historical Period**  \n"
            f"📅 1996 → {max_year}\n\n"
            f"**Sample Size**  \n"
            f"📈 {len(df_raw_ref)} annual observations\n\n"
            f"**Active Model**  \n"
            f"{MODEL_ICONS.get(model_choice, '📊')} {MODEL_LABELS.get(model_choice, 'N/A')}"
        )

    return dict(
        head=head,
        is_multimodel=model_choice in ("ardl", "arimax", "enet"),
        model_choice=model_choice,
        horizon=horizon,
        n_sims=n_sims,
        targets=targets,
        elasticities=elasticities,
        covid_on=covid_on,
        regime_on=regime_on,
        input_mode=input_mode,
        use_demand_supply_shocks=use_demand_supply_shocks,
    )

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

# Load data (initial load - will be reloaded with shock mode if needed)
try:
    if "user_df" in st.session_state:
        df_raw = st.session_state["user_df"]
    else:
        # Initial load without shock decomposition
        # Will be reloaded after sidebar if shock mode is enabled
        df_raw = load_tax_data(decompose_shocks=False)
        df_raw = prepare_transforms(df_raw)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

import hashlib as _hashlib
_data_version = _hashlib.md5(pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()).hexdigest()[:12]

bundle, meta, df_hist_req = load_multimodel_assets()
if "user_df" in st.session_state:
    df_hist = st.session_state["user_df"]
else:
    df_hist = df_hist_req

buoy_data = load_buoyancy()

# ============================================================================
# 2026 ROW PERSISTENCE — username-based, fully persistent
# User types a username once in the sidebar. That name becomes their key.
# Data is saved to user_sessions/<username>.json and reloaded every time
# they open the app and enter the same username. Simple and reliable.
# ============================================================================

_SESSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_sessions")
os.makedirs(_SESSIONS_DIR, exist_ok=True)

def _persist_file(username: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in username.strip().lower())
    return os.path.join(_SESSIONS_DIR, f"user_{safe}.json")

def _save_2026_overrides(username: str) -> None:
    if not username.strip():
        return
    try:
        payload = {}
        if "revised_2026_df" in st.session_state:
            payload["revised_2026_df"] = st.session_state["revised_2026_df"].to_dict(orient="records")
        if "editor_last_year" in st.session_state:
            payload["editor_last_year"] = st.session_state["editor_last_year"].to_dict(orient="records")
        with open(_persist_file(username), "w", encoding="utf-8") as _f:
            json.dump(payload, _f)
    except Exception:
        pass

def _load_2026_overrides(username: str) -> dict:
    if not username.strip():
        return {}
    try:
        path = _persist_file(username)
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as _f:
            raw = json.load(_f)
        result = {}
        for key in ("revised_2026_df", "editor_last_year"):
            if key in raw and raw[key]:
                result[key] = pd.DataFrame(raw[key])
        return result
    except Exception:
        return {}

# ── Username widget — shown at the very top of the sidebar ───────────────────
# We render it here (before the rest of the sidebar) so the username is
# available for all seeding logic below.
_username_input = st.sidebar.text_input(
    "👤 Your Username",
    value=st.session_state.get("_prfs_username", ""),
    placeholder="Enter a username to save your data…",
    key="_prfs_username_widget",
    help="Enter any username. Your 2026 row edits will be saved and reloaded whenever you use the same name.",
)
_username = _username_input.strip()

# Detect username change → reload persisted data for the new user
_prev_username = st.session_state.get("_prfs_username", "")
if _username != _prev_username:
    st.session_state["_prfs_username"] = _username
    # Clear current data so it reloads from the new user's file
    for _k in ("revised_2026_df", "editor_last_year", "user_df"):
        st.session_state.pop(_k, None)
    mm_get_cached_forecast.clear()
    st.rerun()

st.session_state["_prfs_username"] = _username
_persisted = _load_2026_overrides(_username)

# ── Seed 'editor_last_year' at startup (Now Budget Estimates section)
# Source: persisted overrides > Actual 2026 row from Excel (Original Budget)
if "editor_last_year" not in st.session_state and df_raw is not None:
    if "editor_last_year" in _persisted:
        st.session_state["editor_last_year"] = _persisted["editor_last_year"]
    else:
        _log_cols_seed = [c for c in df_raw.columns if c.startswith("log_")]
        _budget_cols_seed = [c for c in df_raw.columns if c not in _log_cols_seed]
        _df_raw_clean = df_raw.reset_index(drop=True)
        if "year_end" in _df_raw_clean.columns and 2026 in _df_raw_clean["year_end"].values:
            _budget_row = _df_raw_clean[_df_raw_clean["year_end"] == 2026][_budget_cols_seed].reset_index(drop=True)
        else:
            _budget_row = _df_raw_clean[_budget_cols_seed].tail(1).copy().reset_index(drop=True)
            if "year_end" in _budget_row.columns: _budget_row["year_end"] = 2026
        st.session_state["editor_last_year"] = _budget_row

# ── Seed 'revised_2026_df' at startup (Now main Revised Estimates row)
# Source: persisted overrides > Manual Revised Estimates for CFY
if "revised_2026_df" not in st.session_state and df_raw is not None:
    if "revised_2026_df" in _persisted:
        st.session_state["revised_2026_df"] = _persisted["revised_2026_df"]
    else:
        _rev_row = df_raw.reset_index(drop=True).tail(1).copy()
        _rev_row = _rev_row.reset_index(drop=True)
        if "year_end" in _rev_row.columns: _rev_row["year_end"] = 2026
        # Populate with current revised estimates
        for _c, _v in [("dt", 6524000), ("gst", 4345000), ("fed", 898000), ("customs", 1312000)]:
            if _c in _rev_row.columns:
                _rev_row[_c] = float(_v)
        st.session_state["revised_2026_df"] = _rev_row

multimodel_ok = bundle is not None
dynamic_ok = dynamic_is_available()

perf = mm_perf_table(meta) if meta else None

# Render sidebar — pass df_raw so Dataset Info block works
cfg = render_sidebar(
    perf=perf,
    dynamic_available=dynamic_ok,
    multimodel_available=multimodel_ok,
    df_raw_ref=df_raw,
)

head = cfg["head"]
horizon = cfg["horizon"]
n_sims = cfg["n_sims"]
targets = cfg["targets"]
elasticities = cfg["elasticities"]
covid_on = cfg["covid_on"]
regime_on = cfg["regime_on"]
use_demand_supply_shocks = cfg.get("use_demand_supply_shocks", False)

# Set active mode label for UI display
active_mode_label = "Demand & Supply Shock-Embedded" if use_demand_supply_shocks else "Baseline"

# Reload data with shock decomposition if shock mode is enabled
if use_demand_supply_shocks and "user_df" not in st.session_state:
    try:
        df_raw = load_tax_data(decompose_shocks=True)
        df_raw = prepare_transforms(df_raw)
        # Recalculate data version hash
        _data_version = _hashlib.md5(pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()).hexdigest()[:12]
    except Exception as e:
        st.warning(f"Shock decomposition failed: {e}. Using baseline mode.")
        use_demand_supply_shocks = False
        active_mode_label = "Baseline"

chosen = cfg["model_choice"]
is_mm = chosen in ("ardl", "arimax", "enet")
is_dsm = chosen == "dynamic"
default_model_label = MODEL_LABELS.get(chosen, chosen.upper())

# Generate forecasts
fore_head = None
fore_total = None
exog_future = None
dyn_results = None

try:
    with st.spinner("🔄 Generating forecasts..."):
        if is_mm:
            fore_head, exog_future = mm_forecast_head(
                bundle, meta, df_hist,
                head, chosen, horizon, n_sims,
                targets, elasticities, covid_on, regime_on, _data_version,
                use_shock_mode=use_demand_supply_shocks
            )
            fore_total = mm_forecast_total(
                bundle, meta, df_hist,
                chosen, horizon, n_sims,
                targets, elasticities, covid_on, regime_on, _data_version,
                use_shock_mode=use_demand_supply_shocks
            )
        else:
            # DSM / Dynamic engine
            if "dyn_pipeline" not in st.session_state:
                st.info("🔄 Dynamic pipeline not yet fitted. Click below to run.")
                if st.button("🚀 Run Dynamic Pipeline", use_container_width=True):
                    with st.spinner("Running 2-Stage Econometric Pipeline…"):
                        dynamic_run_pipeline(df_raw)
                    st.rerun()
                st.stop()

            dyn_results, _, _ = dynamic_run_scenario(df_raw, horizon, targets)
            fore_head = dynamic_to_standard_df(dyn_results, head, horizon)
            fore_total = dynamic_to_standard_df(dyn_results, "total", horizon)

            last_year = int(df_raw.index.max().year)
            years = [last_year + i for i in range(1, horizon + 1)]
            pidx = pd.PeriodIndex(years, freq="Y")
            if len(fore_head):
                fore_head.index = pidx
            if len(fore_total):
                fore_total.index = pidx

except Exception as e:
    st.error(f"Forecast error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Historical series (levels)
y_name = None
hist_level = None

if is_mm and bundle:
    y_name = bundle["models"][head]["spec"]["y"]
    hist_level = np.exp(df_hist[y_name])
    hist_level.name = "Historical"
    total_hist = sum(np.exp(df_hist[f"log_{h}"]) for h in ["dt", "gst", "fed", "customs"])
elif not is_mm:  # DSM / dynamic
    log_col = f"log_{head}"
    if log_col in df_raw.columns:
        hist_level = np.exp(df_raw[log_col].dropna())
    total_hist = sum(
        np.exp(df_raw[f"log_{h}"].dropna())
        for h in ["dt", "gst", "fed", "customs"]
        if f"log_{h}" in df_raw.columns
    )

# Calculate metrics for KPI cards
if (
    fore_total is not None and len(fore_total) > 0
    and total_hist is not None and len(total_hist) > 0
):
    # Projection (NFY) locked to 2027
    total_hist_latest = total_hist.iloc[-1] / 1000
    
    # Robustly find 2027 value in forecast
    _mask_2027 = (fore_total.index.year == 2027) if hasattr(fore_total.index, 'year') else [str(i)[:4] == '2027' for i in fore_total.index]
    if any(_mask_2027):
        total_fore_2027 = fore_total[_mask_2027]["yhat"].iloc[0] / 1000.0
    else:
        total_fore_2027 = fore_total["yhat"].iloc[0] / 1000.0 # Fallback
    
    # Use 2027 for growth metrics
    growth_pct_2027 = ((total_fore_2027 * 1000) / (total_hist_latest * 1000) - 1) * 100
    avg_annual_growth = (((fore_total["yhat"].iloc[-1]) / (total_hist_latest * 1000)) ** (1 / horizon) - 1) * 100

    # --- Category (NFY 2027) logic for consistency
    _cmask_2027 = (fore_head.index.year == 2027) if hasattr(fore_head.index, 'year') else [str(i)[:4] == '2027' for i in fore_head.index]
    if any(_cmask_2027):
        category_forecast_2027 = fore_head[_cmask_2027]["yhat"].iloc[0] / 1000.0
    else:
        category_forecast_2027 = fore_head["yhat"].iloc[0] / 1000.0 # Fallback 
    
    # Revised Estimates (CFY) for the category — from main table
    cat_revised_cfy = hist_level.iloc[-1] / 1000.0 if hist_level is not None and not hist_level.empty else 0.0
    
    # Budget Estimates (CFY) for category — from separate comparison table
    cat_budget_cfy = 0.0
    if "editor_last_year" in st.session_state:
        if head in st.session_state["editor_last_year"].columns:
            cat_budget_cfy = st.session_state["editor_last_year"][head].iloc[0] / 1000.0

    category_growth_2027 = ((category_forecast_2027 / cat_revised_cfy) - 1) * 100 if cat_revised_cfy > 0 else 0.0

    # Variables for Insights Panel (locked to NFY 2027 to match KPIs)
    insight_cat_val = category_forecast_2027
    insight_cat_growth = category_growth_2027
    # Note: total_fore_2027 is already defined above
    
    mae_display = "N/A"
    if perf is not None and chosen in ("ardl", "arimax", "enet"):
        model_mae = perf[(perf["tax_head"] == head) & (perf["model"] == chosen)]["mae_pct"].values
        mae_display = f"{model_mae[0]:.2f}%" if len(model_mae) > 0 else "N/A"
else:
    total_hist_latest = total_fore_2027 = growth_pct_2027 = avg_annual_growth = total_fore_last = 0.0
    category_forecast = category_current = category_growth = 0.0
    mae_display = "N/A"

# ── Helper to sum the 4 tax heads from any single-row DataFrame
def _sum_tax_cols(df):
    total = 0.0
    for c in ["dt", "gst", "fed", "customs"]:
        if c in df.columns:
            val = df[c].iloc[0]
            total += float(val if pd.notnull(val) else 0.0)
    return total

# Revised Estimates (CFY) — reads from main dataset (retrains models)
revised_total_bn = 0.0
if "revised_2026_df" in st.session_state:
    _rdf = st.session_state["revised_2026_df"]
    if not _rdf.empty:
        revised_total_bn = _sum_tax_cols(_rdf) / 1000.0

# Budget Estimates (CFY) — reads from separate comparison section
budget_total_bn = 0.0
if "editor_last_year" in st.session_state:
    _bdf = st.session_state["editor_last_year"]
    if not _bdf.empty:
        budget_total_bn = _sum_tax_cols(_bdf) / 1000.0

# CAGR (%) = ((Projection - Revised) / Revised) * 100
total_cagr_final = 0.0
if revised_total_bn > 0:
    total_cagr_final = ((total_fore_2027 / revised_total_bn) - 1) * 100

# ============================================================================
# HEADER
# ============================================================================
st.markdown(f"""
<div class="executive-header">
    <div class="header-content">
        <div class="header-left">
            <div class="header-title">Pakistan Revenue Forecasting System (PRFS)</div>
            <div class="header-subtitle">ARDL • ARIMAX • ElasticNet • Dynamic Structural Model with bootstrap uncertainty</div>
        </div>
        <div class="header-right">
            <div class="status-badge">
                <div class="status-indicator"></div>
                <span>{MODEL_ICONS.get(chosen, '📊')} {MODEL_LABELS.get(chosen, chosen.upper())}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# KEY METRICS — redesigned cards
# ============================================================================

# Pre-compute deltas
growth_vs_budget  = ((total_fore_2027 / budget_total_bn)  - 1) * 100 if budget_total_bn  > 0 else 0.0
growth_vs_revised = ((total_fore_2027 / revised_total_bn) - 1) * 100 if revised_total_bn > 0 else 0.0

def _chip(val, label, show_sign=True):
    sign   = "+" if val >= 0 else ""
    cls    = "up" if val >= 0 else "down"
    arrow  = "▲" if val >= 0 else "▼"
    txt    = f"{arrow} {sign}{val:.1f}% {label}" if show_sign else f"{arrow} {label}"
    return f'<span class="kpi-chip {cls}">{txt}</span>'

def _chip_flat(label):
    return f'<span class="kpi-chip flat">— {label}</span>'

col1, col2, col3, col4 = st.columns(4)

with col1:
    chip_b = _chip(growth_vs_budget,  "vs Budget")
    chip_r = _chip(growth_vs_revised, "vs Revised")
    st.markdown(f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-top">
                <div class="kpi-icon-pill">💰</div>
                <span class="kpi-badge">NFY Projection</span>
            </div>
            <div class="kpi-label">Total Forecast Revenue</div>
            <div class="kpi-value">Rs. {total_fore_2027:,.1f}B</div>
        </div>
        <div class="kpi-footer">
            <div class="kpi-divider"></div>
            {chip_b}
            {chip_r}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    chip_be = _chip_flat("Current FY")
    st.markdown(f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-top">
                <div class="kpi-icon-pill">🏛️</div>
                <span class="kpi-badge">Budget Est.</span>
            </div>
            <div class="kpi-label">Budget Estimates (CFY)</div>
            <div class="kpi-value">Rs. {budget_total_bn:,.1f}B</div>
        </div>
        <div class="kpi-footer">
            <div class="kpi-divider"></div>
            {chip_be}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    chip_re = _chip_flat("Current FY")
    st.markdown(f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-top">
                <div class="kpi-icon-pill">📋</div>
                <span class="kpi-badge">Revised Est.</span>
            </div>
            <div class="kpi-label">Revised Estimates (CFY)</div>
            <div class="kpi-value">Rs. {revised_total_bn:,.1f}B</div>
        </div>
        <div class="kpi-footer">
            <div class="kpi-divider"></div>
            {chip_re}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    chip_cagr = _chip(total_cagr_final, "vs Revised")
    st.markdown(f"""
    <div class="kpi-card">
        <div>
            <div class="kpi-top">
                <div class="kpi-icon-pill">📈</div>
                <span class="kpi-badge">Growth Rate</span>
            </div>
            <div class="kpi-label">CAGR — NFY vs CFY</div>
            <div class="kpi-value">{total_cagr_final:+.1f}%</div>
        </div>
        <div class="kpi-footer">
            <div class="kpi-divider"></div>
            {chip_cagr}
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 All Categories",
    "📉 Buoyancy vs Model Forecast",
    "🎯 Forecast Accuracy",
    "📋 Model Summary",
    "🔬 Model Diagnostics",
    "📖 Model Guide",
    "💾 Data Preview",
    "🎛️ Sensitivity Analysis",
])

# ─── TAB 1 — All Categories (Plot view) ────────────────────────────────────
with tab1:
    # ─── TOTAL REVENUE (TOP) ──────────────────────────────────────────────────
    if fore_total is not None and len(fore_total):
        fig_total = forecast_plot(total_hist, fore_total, "Total Tax Revenue (Sum of Heads)")
        st.plotly_chart(fig_total, use_container_width=True)
        with st.expander("📊 View Total Forecast Table"):
            st.dataframe(
                forecast_table(fore_total).style.format({"Forecast (PKR Billion)": "{:,.2f}"}),
                use_container_width=True,
            )

    # ─── GRID OVERVIEW ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">All Tax Categories Overview</div>
                <div class="section-subtitle">Individual forecasts using selected model</div>
            </div>
            <div class="section-badge">{MODEL_ICONS.get(chosen, '📊')} {chosen.upper()}</div>
        </div>
    """, unsafe_allow_html=True)

    # Generate forecasts for all categories
    progress_bar_cat = st.progress(0, text="Loading category forecasts...")

    categories = [
        {"key": "customs", "name": "Customs Duty",              "icon": "💰", "icon_class": "customs"},
        {"key": "dt",      "name": "Income / Direct Tax (DT)", "icon": "📋", "icon_class": "dt"},
        {"key": "fed",     "name": "Federal Excise Duty (FED)","icon": "🏭", "icon_class": "fed"},
        {"key": "gst",     "name": "Sales Tax / GST",          "icon": "🛒", "icon_class": "gst"},
    ]

    category_forecasts: Dict = {}
    category_historical: Dict = {}

    for i, cat in enumerate(categories):
        cat_head = cat["key"]
        progress_bar_cat.progress(
            (i + 1) / len(categories),
            text=f"Loading {cat['name']}...",
        )
        try:
            if is_mm and bundle:
                fore_cat, _ = mm_forecast_head(
                    bundle, meta, df_hist,
                    cat_head, chosen, horizon, n_sims,
                    targets, elasticities, covid_on, regime_on, _data_version,
                    use_shock_mode=use_demand_supply_shocks
                )
                y_name_cat = bundle["models"][cat_head]["spec"]["y"]
                hist_cat = np.exp(df_hist[y_name_cat])
            elif not is_mm and dyn_results:
                fore_cat = dynamic_to_standard_df(dyn_results, cat_head, horizon)
                log_col = f"log_{cat_head}"
                hist_cat = np.exp(df_raw[log_col].dropna()) if log_col in df_raw.columns else pd.Series()
                last_year = int(df_raw.index.max().year)
                years_cat = [last_year + i for i in range(1, horizon + 1)]
                pidx_cat = pd.PeriodIndex(years_cat, freq="Y")
                if len(fore_cat):
                    fore_cat.index = pidx_cat
            else:
                continue

            category_forecasts[cat_head] = fore_cat
            category_historical[cat_head] = hist_cat
        except Exception as e:
            st.warning(f"Could not generate forecast for {cat['name']}: {e}")
            continue

    progress_bar_cat.empty()

    # Render category cards in 2-column grid
    for i in range(0, len(categories), 2):
        cols = st.columns(2, gap="large")
        for j in range(2):
            if i + j >= len(categories):
                break
            cat = categories[i + j]
            cat_head = cat["key"]
            if cat_head not in category_forecasts:
                continue

            fore_cat = category_forecasts[cat_head]
            hist_cat = category_historical[cat_head]

            with cols[j]:
                # Metrics
                if len(hist_cat) > 0 and len(fore_cat) > 0:
                    # Revised (CFY) from main table
                    cat_revised = (st.session_state["revised_2026_df"][cat_head].iloc[0] / 1000.0) if cat_head in st.session_state["revised_2026_df"].columns else (hist_cat.iloc[-1] / 1000.0)
                    # Budget (CFY) from separate table
                    cat_budget = (st.session_state["editor_last_year"][cat_head].iloc[0] / 1000.0) if cat_head in st.session_state["editor_last_year"].columns else (hist_cat.iloc[-1] / 1000.0)
                    
                    # Projection (NFY) locked to 2027
                    cat_projection_2027 = 0.0
                    if len(fore_cat) > 0:
                        _cmask_2027 = (fore_cat.index.year == 2027) if hasattr(fore_cat.index, 'year') else [str(i)[:4] == '2027' for i in fore_cat.index]
                        if any(_cmask_2027):
                            cat_projection_2027 = fore_cat[_cmask_2027]["yhat"].iloc[0] / 1000.0
                        else:
                            cat_projection_2027 = fore_cat["yhat"].iloc[0] / 1000.0
                    
                    # CAGR (%) = ((Projection - Revised) / Revised) * 100
                    cat_cagr_new = ((cat_projection_2027 / cat_revised) - 1) * 100 if cat_revised > 0 else 0.0
                    
                    cat_mae = "N/A"
                    if perf is not None and chosen in ("ardl", "arimax", "enet"):
                        cat_mae_vals = perf[(perf["tax_head"] == cat_head) & (perf["model"] == chosen)]["mae_pct"].values
                        cat_mae = f"{cat_mae_vals[0]:.2f}%" if len(cat_mae_vals) > 0 else "N/A"
                else:
                    cat_budget = cat_revised = cat_projection_2027 = cat_cagr_new = 0.0
                    cat_mae = "N/A"

                st.markdown(f"""
                <div class="category-card">
                    <div class="category-header">
                        <div class="category-icon {cat['icon_class']}">{cat['icon']}</div>
                        <div>
                            <div class="category-title">{cat['name']}</div>
                        </div>
                    </div>
                    <div class="category-metrics">
                        <div class="metric-item">
                            <div class="metric-label">Revised Estimates (CFY)</div>
                            <div class="metric-value">₨{cat_revised:,.1f}B</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Budget Estimates (CFY)</div>
                            <div class="metric-value">₨{cat_budget:,.1f}B</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Projection (NFY)</div>
                            <div class="metric-value">₨{cat_projection_2027:,.1f}B</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">CAGR (%)</div>
                            <div class="metric-value">{cat_cagr_new:+.1f}%</div>
                        </div>
                    </div>
                    <div class="model-info-badge">
                        <span><strong>Model:</strong> {MODEL_LABELS.get(chosen, chosen.upper())}</span>
                        <span class="mae-badge">MAE: {cat_mae}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Mini chart
                if len(hist_cat) > 0 and len(fore_cat) > 0:
                    fig_cat = go.Figure()
                    x_hist_cat = hist_cat.index.map(lambda p: pd.Timestamp(year=p.year, month=6, day=30))
                    x_fore_cat = fore_cat.index.map(lambda p: pd.Timestamp(year=p.year, month=6, day=30))

                    fig_cat.add_trace(go.Scatter(
                        x=x_hist_cat, y=hist_cat.values / 1000,
                        mode="lines+markers", name="Historical / Budget",
                        line=dict(color="#2563EB", width=3),
                        marker=dict(size=6, color="#2563EB", line=dict(width=1, color="white")),
                        fill="tozeroy", fillcolor="rgba(37, 99, 235, 0.06)",
                    ))

                    # ── Add 'Revised' marker if available for visual context
                    if "editor_last_year" in st.session_state and cat_head in st.session_state["editor_last_year"].columns:
                        _rev_val = st.session_state["editor_last_year"][cat_head].iloc[0] / 1000.0
                        if _rev_val > 0:
                            fig_cat.add_trace(go.Scatter(
                                x=[pd.Timestamp(year=2026, month=6, day=30)],
                                y=[_rev_val],
                                mode="markers",
                                name="Revised Est. (CFY)",
                                marker=dict(symbol="diamond", size=10, color="#F59E0B", line=dict(width=2, color="white")),
                                hovertemplate="Revised 2026: Rs. %{y:,.1f}B<extra></extra>"
                            ))
                    fig_cat.add_trace(go.Scatter(
                        x=np.concatenate([x_fore_cat, x_fore_cat[::-1]]),
                        y=np.concatenate([fore_cat["hi95"] / 1000, fore_cat["lo95"][::-1] / 1000]),
                        fill="toself", fillcolor="rgba(139, 92, 246, 0.1)",
                        line=dict(color="rgba(255,255,255,0)"), showlegend=False,
                    ))
                    fig_cat.add_trace(go.Scatter(
                        x=np.concatenate([x_fore_cat, x_fore_cat[::-1]]),
                        y=np.concatenate([fore_cat["hi80"] / 1000, fore_cat["lo80"][::-1] / 1000]),
                        fill="toself", fillcolor="rgba(139, 92, 246, 0.2)",
                        line=dict(color="rgba(255,255,255,0)"), showlegend=False,
                    ))
                    fig_cat.add_trace(go.Scatter(
                        x=x_fore_cat, y=fore_cat["yhat"] / 1000,
                        mode="lines+markers", name="Forecast",
                        line=dict(color="#8B5CF6", width=3, dash="dash"),
                        marker=dict(size=7, color="#8B5CF6", line=dict(width=2, color="white")),
                    ))
                    fig_cat.update_layout(
                        height=250,
                        margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        showlegend=False,
                        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", title="Year", title_font=dict(size=10)),
                        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", title="PKR Billion", title_font=dict(size=10)),
                    )
                    st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False, "displaylogo": False})

    st.markdown("</div>", unsafe_allow_html=True)  # close content-section

# ─── TAB 2 — Buoyancy vs Model Forecast ────────────────────────────────────
with tab2:
    st.markdown(f"""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Buoyancy vs Model Forecast Comparison</div>
                <div class="section-subtitle">Evaluating model output against historical revenue-buoyancy benchmarks</div>
            </div>
            <div class="section-badge">📊 Comparative Analysis</div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Load Buoyancy Reference Benchmarks
    b_ref = load_buoyancy()
    
    # 2. Setup Comparison Parameters
    b_map = {"customs": "customs", "dt": "dt", "fed": "fed", "gst": "st_total"}
    head_names = {
        "customs": "Customs Duty (CD)",
        "dt": "Direct Taxes (DT)",
        "fed": "Federal Excise Duty (FED)",
        "gst": "Sales Tax (GST)"
    }
    
    model_keys = [
        ("ardl", "ADRL"), 
        ("arimax", "ARIMAX"), 
        ("enet", "ElasticNet"),
        ("dynamic", "Dynamic Structural Model")
    ]
    
    buoy_rows = []
    
    def _extract_fy27_metrics_tab(f_df):
        if f_df is None or f_df.empty: return None
        m27 = (f_df.index.year == 2027) if hasattr(f_df.index, 'year') else [str(i)[:4] == '2027' for i in f_df.index]
        if any(m27):
            row = f_df[m27].iloc[0]
            return row["yhat"]/1000.0, row["lo95"]/1000.0, row["hi95"]/1000.0
        return f_df["yhat"].iloc[0]/1000.0, f_df["lo95"].iloc[0]/1000.0, f_df["hi95"].iloc[0]/1000.0

    with st.spinner("Calculating comparison metrics..."):
        d_res_tab = None
        if "dyn_pipeline" in st.session_state:
            try: d_res_tab, _, _ = dynamic_run_scenario(df_raw, horizon, targets)
            except Exception: pass

        for m_key, m_label in model_keys:
            if m_key == "dynamic" and d_res_tab is None: continue
            if m_key != "dynamic" and not bundle: continue

            for h_key, h_label in head_names.items():
                f26_b = f27_b = 0.0
                if b_ref:
                    b_col = b_map.get(h_key)
                    f26_b = b_ref.get("fy2026_base", {}).get(b_col, 0.0)
                    f27_b = b_ref.get("fy2027_buoyancy", {}).get(b_col, 0.0)

                m_proj_val = 0.0
                m_ci_text = "N/A"
                
                try:
                    if m_key == "dynamic": f_df = dynamic_to_standard_df(d_res_tab, h_key, horizon)
                    else:
                        f_df, _ = mm_forecast_head(bundle, meta, df_hist, h_key, m_key, horizon, n_sims, targets, elasticities, covid_on, regime_on, _data_version, use_shock_mode=use_demand_supply_shocks)
                    
                    metrics = _extract_fy27_metrics_tab(f_df)
                    if metrics:
                        val27, lo, hi = metrics
                        m_proj_val = val27
                        m_ci_text = f"{lo:,.1f} — {hi:,.1f}"
                except Exception: pass

                buoy_rows.append({
                    "Model": m_label, "Tax Head": h_label,
                    "FY26 Buoyancy (bn)": f26_b, "FY27 Buoyancy (bn)": f27_b,
                    "FY27 Macro Model (bn)": m_proj_val, "95% CI (bn)": m_ci_text
                })

    if buoy_rows:
        df_buoy_tab = pd.DataFrame(buoy_rows)
        df_buoy_tab = df_buoy_tab[["Model", "Tax Head", "FY26 Buoyancy (bn)", "FY27 Buoyancy (bn)", "FY27 Macro Model (bn)", "95% CI (bn)"]]
        st.dataframe(df_buoy_tab.style.format({"FY26 Buoyancy (bn)": "{:,.1f}", "FY27 Buoyancy (bn)": "{:,.1f}", "FY27 Macro Model (bn)": "{:,.1f}"}), use_container_width=True, hide_index=True)
        csv_buoy = df_buoy_tab.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Comparison (CSV)", data=csv_buoy, file_name="prfs_buoyancy_vs_model_comparison.csv", mime="text/csv")
    else: st.warning("⚠️ Comparison data unavailable.")

    st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3 - Forecast Accuracy ---
with tab3:
    st.markdown(f"""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Forecast Accuracy</div>
                <div class="section-subtitle">
                    Out-of-sample backtest — <strong>n_test={N_BT}</strong> folds (same folds
                    used for the accuracy table, model ranking, and the graph below).
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if is_mm and bundle and df_hist is not None:

        # ── Head / model selectors ────────────────────────────────────────────
        bt_col1, bt_col2 = st.columns([1, 1])
        with bt_col1:
            bt_head = st.selectbox(
                "Tax head",
                options=list(TAX_LABELS.keys()),
                format_func=lambda x: TAX_LABELS[x],
                index=list(TAX_LABELS.keys()).index(head),
                key="bt_head_select",
            )

        # ── Run ALL models with SAME n_test — single source of truth ─────────
        with st.spinner(f"Running OOS backtest ({N_BT} folds) for all models…"):
            _all_bt = build_all_model_backtests(bt_head, bundle, df_hist, n_test=N_BT)

        # Derive best model from live backtest (not from stored meta)
        _live_best = best_model_from_backtest_results(_all_bt)

        with bt_col2:
            _model_opts = [m for m in ("ardl", "arimax", "enet", "dsm") if m in _all_bt]
            if not _model_opts:
                _model_opts = ["ardl"]
            _default_idx = _model_opts.index(_live_best) if _live_best in _model_opts else 0
            bt_model = st.selectbox(
                "Model (★ = best stable by average OOS sMAPE)",
                options=_model_opts,
                format_func=lambda x: f"★ {x.upper()} (Best)" if x == _live_best else x.upper(),
                index=_default_idx,
                key="bt_model_select",
            )

        # ── Accuracy Table — built from LIVE OOS backtest metrics ────────────
        # ── Best-model recommendation banner ─────────────────────────────────
        _best_label = MODEL_LABELS.get(_live_best, _live_best.upper())
        _best_icon  = MODEL_ICONS.get(_live_best, "📊")
        _sidebar_key = "dynamic" if _live_best == "dsm" else _live_best
        st.success(
            f"**{_best_icon} OOS Best Model for {TAX_LABELS[bt_head]}: {_best_label}** "
            f"(lowest average OOS sMAPE across all stable models, RMSE% as tiebreaker). "
            f"To use this model for the final forecast, select **'{MODEL_LABELS.get(_sidebar_key, _sidebar_key)}'** "
            f"in the sidebar. The final forecast is always produced on the full historical sample."
        )
        if _live_best in ("enet", "arimax"):
            st.info(
                "ℹ️ **Sensitivity Analysis Note:** Even when ElasticNet or ARIMAX win the forecast "
                "accuracy ranking, the Sensitivity Analysis tab uses ARDL or DSM "
                "structural coefficients for policy interpretation — not the pure accuracy winner."
            )

        st.markdown("#### 📋 Forecast Accuracy — OOS Backtest (All Models, Same Folds)")
        _acc_rows = []
        _MODEL_DISPLAY_ORDER = ("ardl", "arimax", "enet", "dsm")
        for _mk in _MODEL_DISPLAY_ORDER:
            if _mk not in _all_bt:
                continue
            _act_s, _oos_df = _all_bt[_mk]
            _unstable = _is_unstable(_oos_df, _act_s)
            _m = compute_metrics_from_backtest(_act_s, _oos_df)

            if _unstable:
                _status = "⚠️ Unstable"
                _label = f"⚠️ {_mk.upper()} (Unstable)"
            elif _mk == _live_best:
                _status = "★ Best"
                _label = f"★ {_mk.upper()} (Best)"
            else:
                _status = "✓ Stable"
                _label = _mk.upper()

            _acc_rows.append({
                "Model":         _label,
                "Status":        _status,
                "OOS sMAPE%":    _m["smape_all"] if not _unstable else np.nan,
                "h3 sMAPE%":     _m["h3_smape"]  if not _unstable else np.nan,
                "RMSE%":         _m["rmse_pct"]  if not _unstable else np.nan,
                "h1 (1yr, ref)": _m["h1_smape"]  if not _unstable else np.nan,
                "n_test":        _m["n"],
            })

        if _acc_rows:
            _acc_df = pd.DataFrame(_acc_rows)
            # Sort by primary metric (avg OOS sMAPE), then RMSE% as tiebreaker
            _acc_df["_s1"] = _acc_df["OOS sMAPE%"].fillna(9999.0)
            _acc_df["_s2"] = _acc_df["RMSE%"].fillna(9999.0)
            _acc_df = (_acc_df.sort_values(["_s1", "_s2"])
                       .drop(columns=["_s1", "_s2"]).reset_index(drop=True))
            st.dataframe(
                _acc_df.style.format({
                    "OOS sMAPE%":    lambda v: f"{v:.2f}%" if pd.notnull(v) else "—",
                    "h3 sMAPE%":     lambda v: f"{v:.2f}%" if pd.notnull(v) else "—",
                    "RMSE%":         lambda v: f"{v:.2f}%" if pd.notnull(v) else "—",
                    "h1 (1yr, ref)": lambda v: f"{v:.2f}%" if pd.notnull(v) else "—",
                    "n_test":        "{:.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"★ = Best stable model (lowest average OOS sMAPE, RMSE% as tiebreaker). "
                f"⚠️ = Unstable (NaN/Inf/negative/explosive). "
                f"All metrics computed from the **same {N_BT}-fold OOS backtest** used for the graph below. "
                f"Unstable models are **excluded** from best-model selection."
            )
            st.markdown("""
**Metric Guide:**
| Metric | Meaning |
|--------|---------|
| **OOS sMAPE%** | Symmetric MAPE averaged over **all** OOS folds — **primary ranking metric** |
| **h3 sMAPE%** | Average sMAPE over the last 3 OOS folds — supporting metric |
| **RMSE%** | Root-mean-square error as % of mean actual — **tiebreaker** |
| **h1 (1yr, ref)** | sMAPE on the single most recent fold — shown for reference only, *not* used for ranking (one year is too noisy to rank on) |
| **n_test** | OOS prediction folds used |
| **Status** | ★ Best / ✓ Stable / ⚠️ Unstable (excluded from ranking) |
""")
        else:
            st.info("No backtest data available. Ensure multi-model bundle is loaded.")

        st.markdown("---")

        # ── Actual vs Predicted Graph — same backtest data as table ──────────
        st.markdown("#### 📈 Actual vs Predicted (OOS Backtest — Same Folds as Accuracy Table)")
        st.caption(
            f"The graph uses the **exact same {N_BT}-fold OOS backtest** as the accuracy table above. "
            f"Metrics recalculated from the graph must match the table."
        )

        if bt_model in _all_bt:
            bt_actual, bt_oos = _all_bt[bt_model]
        else:
            bt_actual, bt_oos = pd.Series(dtype=float), pd.DataFrame()

        if len(bt_actual) > 0:
            fig_bt = go.Figure()

            fig_bt.add_trace(go.Scatter(
                x=list(bt_actual.index),
                y=(bt_actual / 1000).tolist(),
                mode="lines+markers",
                name="Actual",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=5, color="#2563EB"),
            ))

            if len(bt_oos) > 0:
                _bt_label = bt_model.upper()
                _bt_status = " ⚠️Unstable" if _is_unstable(bt_oos, bt_actual) else ""
                fig_bt.add_trace(go.Scatter(
                    x=bt_oos["year"].tolist(),
                    y=(bt_oos["yhat"] / 1000).tolist(),
                    mode="lines+markers",
                    name=f"OOS Predicted ({_bt_label}, n={N_BT}){_bt_status}",
                    line=dict(color="#F59E0B", width=2.2, dash="dash"),
                    marker=dict(symbol="x", size=9, color="#F59E0B",
                                line=dict(width=2, color="#F59E0B")),
                ))

            fig_bt.update_layout(
                title=dict(
                    text=(
                        f"{TAX_LABELS[bt_head]} — Actual vs {bt_model.upper()} "
                        f"OOS Backtest ({N_BT} folds)"
                    ),
                    font=dict(size=14, color="#1F2937"), x=0,
                ),
                xaxis=dict(title="Fiscal Year", showgrid=True,
                           gridcolor="rgba(0,0,0,0.06)", tickformat="d"),
                yaxis=dict(title="PKR bn", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                height=420, margin=dict(l=10, r=10, t=60, b=40),
            )
            st.plotly_chart(fig_bt, use_container_width=True,
                            config={"displayModeBar": False})

            # ── Post-plot verification: recalculate from plotted data ─────────
            _graph_m = compute_metrics_from_backtest(bt_actual, bt_oos)
            _g_all   = _graph_m.get("smape_all", np.nan)
            _g_h3    = _graph_m.get("h3_smape", np.nan)
            _g_rmse  = _graph_m.get("rmse_pct", np.nan)

            # Find corresponding table row for this model (any model name)
            _tbl_row = next(
                (r for r in _acc_rows if bt_model.upper() in r["Model"].upper()), None
            )
            _t_all = _tbl_row["OOS sMAPE%"] if _tbl_row and pd.notnull(_tbl_row["OOS sMAPE%"]) else np.nan

            # Flag instability in the info display
            _unstable_flag = _is_unstable(bt_oos, bt_actual)
            _all_str  = f"{_g_all:.2f}%"  if not np.isnan(_g_all)  else "N/A"
            _h3_str   = f"{_g_h3:.2f}%"   if not np.isnan(_g_h3)   else "N/A"
            _rmse_str = f"{_g_rmse:.2f}%" if not np.isnan(_g_rmse) else "N/A"

            if _unstable_flag:
                st.warning(
                    f"⚠️ **{bt_model.upper()} flagged as UNSTABLE** for this tax head. "
                    f"It has been excluded from best-model ranking. "
                    f"OOS sMAPE: **{_all_str}** · h3 sMAPE: **{_h3_str}** · RMSE%: **{_rmse_str}** "
                    f"(n = {_graph_m['n']} OOS folds)"
                )
            else:
                st.info(
                    f"**Metrics recalculated from the graph above** — "
                    f"OOS sMAPE: **{_all_str}** · h3 sMAPE: **{_h3_str}** · RMSE%: **{_rmse_str}**  "
                    f"(n = {_graph_m['n']} OOS folds)"
                )

            # Warn if live graph metrics diverge from table by >1 pp (skip if unstable)
            if not _unstable_flag:
                if not np.isnan(_g_all) and not np.isnan(_t_all):
                    if abs(_g_all - _t_all) > 1.0:
                        st.warning(
                            f"⚠️ **Consistency check failed**: graph OOS sMAPE ({_g_all:.2f}%) "
                            f"differs from table value ({_t_all:.2f}%) by more than 1 pp. "
                            f"This should not happen — please report this as a bug."
                        )
                    else:
                        st.success("✅ Consistency check passed: graph metrics match table values.")
        else:
            st.info("No backtest data available for this combination.")

        _mm_only_models = ("ardl", "arimax", "enet")  # used by the Forecast Values Table below

        st.markdown("---")

        # ── Forecast Values Table ─────────────────────────────────────────────
        st.markdown("#### 📋 Forecast Values Table")
        _fore_tbl_cols = st.columns([1, 1])
        with _fore_tbl_cols[0]:
            ft_head = st.selectbox(
                "Tax head",
                options=list(TAX_LABELS.keys()),
                format_func=lambda x: TAX_LABELS[x],
                index=list(TAX_LABELS.keys()).index(bt_head),
                key="ft_head_select",
            )
        with _fore_tbl_cols[1]:
            _ft_model_opts = list(_mm_only_models)
            _ft_default = bt_model if bt_model in _ft_model_opts else "ardl"
            ft_model = st.selectbox(
                "Model",
                options=_ft_model_opts,
                format_func=lambda x: x.upper(),
                index=_ft_model_opts.index(_ft_default),
                key="ft_model_select",
            )

        with st.spinner("Loading forecast values…"):
            try:
                ft_fore_df, _ = mm_forecast_head(
                    bundle, meta, df_hist,
                    ft_head, ft_model, horizon, n_sims,
                    targets, elasticities, covid_on, regime_on, _data_version,
                    use_shock_mode=use_demand_supply_shocks
                )
            except Exception:
                ft_fore_df = None

        if ft_fore_df is not None and len(ft_fore_df) > 0:
            _ft_years = [p.year if hasattr(p, "year") else int(str(p)[:4]) for p in ft_fore_df.index]
            _ft_rows = []
            for yr, (_, row) in zip(_ft_years, ft_fore_df.iterrows()):
                _ft_rows.append({
                    "Fiscal Year":       yr,
                    "Forecast (PKR bn)": round(row["yhat"] / 1000, 2),
                    "80% CI Low":        round(row["lo80"] / 1000, 2),
                    "80% CI High":       round(row["hi80"] / 1000, 2),
                    "95% CI Low":        round(row["lo95"] / 1000, 2),
                    "95% CI High":       round(row["hi95"] / 1000, 2),
                })
            ft_display = pd.DataFrame(_ft_rows)

            st.markdown(f"**Out-of-Sample Forecast — {TAX_LABELS[ft_head]}**")
            st.dataframe(
                ft_display.style.format({
                    "Forecast (PKR bn)": "{:,.2f}",
                    "80% CI Low": "{:,.2f}", "80% CI High": "{:,.2f}",
                    "95% CI Low": "{:,.2f}", "95% CI High": "{:,.2f}",
                }),
                use_container_width=True, hide_index=True,
            )

            csv_ft = ft_display.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast Table (CSV)",
                data=csv_ft,
                file_name=f"prfs_forecast_{ft_head}_{ft_model}.csv",
                mime="text/csv",
            )
        else:
            st.info("Forecast values not available. Ensure multi-model bundle is loaded.")

    else:
        st.info("No accuracy metrics available. Load multi-model bundle or run DSM pipeline.")



# ─── TAB 4 — Model Summary ─────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Model Summary</div>
                <div class="section-subtitle">Coefficient estimates and structural parameters</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.write(f"**{TAX_LABELS[head]}** · Engine: **{default_model_label}**")

    if is_mm and bundle:
        head_bundle = bundle["models"][head]
        if chosen == "ardl":
            res = head_bundle["ardl"]["res"]
            st.markdown("### 📊 Coefficient Estimates")
            st.dataframe(
                coef_table_ardl(res).style.format({"coef": "{:.4f}", "std_err": "{:.4f}", "p": "{:.4f}"}),
                use_container_width=True,
            )
            vals = res.params
            y_n = head_bundle["spec"]["y"]
            rho_sum = sum(vals[vals.index.str.startswith(f"{y_n}.L")])
            denom = 1.0 - rho_sum
            lr_rows = []
            
            # Find actual exogenous variables present in the ARDL model
            actual_x = []
            for k in vals.index:
                if k in ("const", "intercept") or k.startswith(f"{y_n}.L"):
                    continue
                col = k.split(".L")[0]
                if col not in actual_x:
                    actual_x.append(col)
                    
            for xc in actual_x:
                gs = sum(vals[vals.index.str.startswith(f"{xc}.L")]) if any(k.startswith(f"{xc}.L") for k in vals.index) else vals.get(xc, 0.0)
                lr_rows.append({"variable": xc, "elasticity": gs / denom if abs(denom) > 1e-4 else 0})

            st.markdown("### 📈 Long-Run Elasticities (ECM)")
            st.markdown(f"**Error Correction Speed:** `{rho_sum - 1.0:.4f}`")
            st.dataframe(pd.DataFrame(lr_rows).style.format({"elasticity": "{:.3f}"}), use_container_width=True)
            with st.expander("📋 Full Model Output"):
                st.text(res.summary().as_text())

        elif chosen == "arimax":
            res = head_bundle["arimax"]["res"]
            st.markdown("### 📊 Coefficient Estimates")
            st.dataframe(
                coef_table_arimax(res).style.format({"coef": "{:.4f}", "std_err": "{:.4f}", "z": "{:.2f}", "p": "{:.4f}"}),
                use_container_width=True,
            )
            with st.expander("📋 Full Model Output"):
                st.text(res.summary().as_text())

        elif chosen == "enet":
            st.markdown("### 📊 ElasticNet Coefficients")
            st.dataframe(
                coef_table_enet(head_bundle).style.format({"coef": "{:.6f}"}),
                use_container_width=True,
            )
            st.markdown("### ⚙️ Model Settings")
            st.json(head_bundle["enet"].get("params", {}))
    else:
        pipeline = st.session_state.get("dyn_pipeline")
        if pipeline and head in pipeline.best_models:
            m = pipeline.best_models[head].get("policy_winner")
            if m and m.elasticities:
                st.markdown("### 📊 Policy Elasticities")
                st.markdown(f"**Winning Model:** {m.name}")
                c1, c2 = st.columns(2)
                c1.markdown("#### Short-Run")
                sr = m.elasticities.get("Short-Run", {})
                if sr:
                    sr_rows = [{"Variable": k, "Description": VAR_GLOSSARY.get(k, (k, ""))[0], "Elasticity": v} for k, v in sr.items()]
                    c1.dataframe(pd.DataFrame(sr_rows).style.format({"Elasticity": "{:.4f}"}), use_container_width=True)
                c2.markdown("#### Long-Run")
                lr = m.elasticities.get("Long-Run", {})
                if lr:
                    lr_rows = [{"Variable": k, "Description": VAR_GLOSSARY.get(k, (k, ""))[0], "Elasticity": v} for k, v in lr.items()]
                    c2.dataframe(pd.DataFrame(lr_rows).style.format({"Elasticity": "{:.4f}"}), use_container_width=True)
            else:
                st.info("No policy elasticities available for this tax head.")
        else:
            st.info("Run the Dynamic Pipeline first to see DSM model summaries.")

    st.markdown("</div>", unsafe_allow_html=True)  # close content-section

# ─── TAB 5 — Model Diagnostics ─────────────────────────────────────────────
with tab5:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Model Diagnostics</div>
                <div class="section-subtitle">Statistical tests and residual analysis</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.write(f"**{TAX_LABELS[head]}** · Engine: **{default_model_label}**")

    if is_mm and bundle:
        head_bundle = bundle["models"][head]
        if chosen == "ardl":
            res = head_bundle["ardl"]["res"]
            diag = diagnostics_ardl(res)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Durbin–Watson", f"{diag['durbin_watson']:.2f}")
            c2.metric("Ljung–Box p", "NA" if diag["ljung_box_p"] is None else f"{diag['ljung_box_p']:.3f}")
            c3.metric("Jarque–Bera p", "NA" if diag["jb_full_p"] is None else f"{diag['jb_full_p']:.3f}")
            c4.metric("JB trimmed p", "NA" if diag["jb_trim_p"] is None else f"{diag['jb_trim_p']:.3f}")
            c5.metric("Breusch–Pagan p", "NA" if diag["breusch_pagan_p"] is None else f"{diag['breusch_pagan_p']:.3f}")
            st.caption("'JB trimmed p' excludes first residual to avoid burn-in artifacts.")
            resid = pd.Series(res.resid).dropna()
            ridx = df_hist.index[-len(resid):]
            resid.index = ridx
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=resid.index.to_timestamp(), y=resid.values,
                mode="lines+markers", name="Residuals",
                line=dict(color="#8B5CF6", width=2), marker=dict(size=6),
            ))
            fig_r.update_layout(
                xaxis_title="Year", yaxis_title="Residual", template="plotly_white",
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.03)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.03)", zeroline=True),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        elif chosen == "arimax":
            res = head_bundle["arimax"]["res"]
            diag = diagnostics_arimax(res)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("AIC", f"{diag['aic']:.1f}")
            c2.metric("Durbin–Watson", f"{diag['durbin_watson']:.2f}")
            c3.metric("Ljung–Box p", "NA" if diag["ljung_box_p"] is None else f"{diag['ljung_box_p']:.3f}")
            c4.metric("Jarque–Bera p", "NA" if diag["jb_full_p"] is None else f"{diag['jb_full_p']:.3f}")
            c5.metric("JB trimmed p", "NA" if diag["jb_trim_p"] is None else f"{diag['jb_trim_p']:.3f}")
            resid = pd.Series(res.resid).dropna()
            ridx = df_hist.index[-len(resid):]
            resid.index = ridx
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=resid.index.to_timestamp(), y=resid.values,
                mode="lines+markers", name="Residuals",
                line=dict(color="#8B5CF6", width=2), marker=dict(size=6),
            ))
            fig_r.update_layout(
                xaxis_title="Year", yaxis_title="Residual", template="plotly_white",
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.03)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.03)", zeroline=True),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        elif chosen == "enet":
            st.write("ElasticNet diagnostics focus on stability and backtest metrics.")
            st.dataframe(
                coef_table_enet(head_bundle).head(15).style.format({"coef": "{:.6f}"}),
                use_container_width=True,
            )
            if perf is not None:
                st.markdown("### 📊 Backtest Performance")
                st.dataframe(
                    perf[perf["tax_head"] == head]
                    .sort_values("mae_pct")[["model", "mae_pct", "rmse_pct", "n_test"]]
                    .style.format({"mae_pct": "{:.2f}%", "rmse_pct": "{:.2f}%"}),
                    use_container_width=True,
                )
    else:
        pipeline = st.session_state.get("dyn_pipeline")
        if pipeline and pipeline.leaderboard:
            st.write("**DSM Tournament Diagnostics** — Expanding-window backtest with recursive forecasting")
            lb = pd.DataFrame(pipeline.leaderboard)
            diag_cols = [c for c in lb.columns if c != "obj"]
            head_lb = lb[lb["Tax Head"] == head.upper()][diag_cols].copy()
            if not head_lb.empty:
                sort_col_diag = "h1_sMAPE%" if "h1_sMAPE%" in head_lb.columns else "sMAPE%"
                if sort_col_diag in head_lb.columns:
                    head_lb = (
                        head_lb.sort_values(sort_col_diag, ascending=True)
                        .drop_duplicates(subset=["Model"], keep="first")
                    )
                show_cols = [c for c in ["Model", "h1_sMAPE%", "h3_sMAPE%", "RMSE%", "n_test"] if c in head_lb.columns]
                if not show_cols:
                    show_cols = [c for c in diag_cols if c in head_lb.columns]
                st.dataframe(head_lb[show_cols], use_container_width=True)
                n_values = head_lb["n_test"].unique() if "n_test" in head_lb.columns else []
                if len(n_values) == 1:
                    st.success(f"✅ Window integrity verified — all models used {int(n_values[0])} identical test origins.")
                elif len(n_values) > 1:
                    st.warning(f"⚠️ Test window mismatch detected: {sorted(n_values)}")
            else:
                st.info(f"No leaderboard entries for {TAX_LABELS[head]}.")
        else:
            st.info("Run the Dynamic Pipeline to see diagnostics.")

    st.markdown("</div>", unsafe_allow_html=True)  # close content-section

# ─── TAB 6 — Model Guide ───────────────────────────────────────────────────
with tab6:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Model Guide</div>
                <div class="section-subtitle">Technical methodology and interpretation</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Technical methodology for each forecasting model employed in this system. "
        "All models operate on **log-transformed, annual** macro-fiscal data for Pakistan (FY1996–FY2026). "
        "The DSM engine uses a **two-stage structural identification** strategy; "
        "the Multi-Model engine runs ARDL, ARIMAX, and Elastic Net in parallel."
    )
    st.markdown("---")

    with st.expander("🏗️ DSM — Two-Stage Dynamic Structural Model (Overview)", expanded=True):
        st.markdown("""
### Motivation
Direct regression of tax revenue on macro aggregates (GDP, imports) suffers from **simultaneity bias**: 
tax policy itself affects GDP and imports, creating a feedback loop that violates the OLS exogeneity assumption. 
The DSM resolves this via a **two-stage instrumental-variable-style** approach analogous to 2SLS, 
where Stage-1 fitted values serve as instruments for the endogenous regressors in Stage-2.

---

### Stage 1 — Channel Equations (Structural Macro Block)

Four **channel equations** are estimated by ARDL on the **full macro panel** to recover 
the *exogenous* component of each intermediate variable:

| Channel | Equation | Rationale |
|---------|----------|-----------|
| **Imports** | `log_imports = f(log_gdp, log_exrate, policy_rate, inflation)` | Demand-side import function; driven by income and price effects |
| **Dutiable Imports** | `log_dutiable_imports = f(log_imports, log_exrate, policy_rate, inflation)` | Composition of import basket subject to tariff |
| **LSM (Large-Scale Manufacturing)** | `log_lsm = f(log_gdp, policy_rate, inflation)` | Output proxy for domestic value-added tax base |
| **GDP** | `log_gdp` | Overall economic driver; primary GST/sales tax base |

Each channel equation is fitted using `ardl_select_order()` with AIC-optimal lag selection, 
subject to the constraint `p, q ≤ n/8` to preserve degrees of freedom on the ~30-observation sample.

**Output:** Fitted values `ŷ` from each equation are stored as:
- `log_imports_hat`
- `log_dutiable_imports_hat`
- `log_lsm_hat`
- `log_gdp_hat` ≡ `log_gdp` (exogenous by assumption)

These `_hat` regressors are **purged of endogeneity** — they represent only the variation in 
the economic base that is explained by exogenous drivers (exchange rate movements, monetary policy, GDP shocks), 
not by reverse causation from the tax administration itself.

---

### Stage 2 — Tax Revenue Equations (Structural Tax Block)

Each tax head is then regressed on its appropriate structural tax base using the Stage-1 fitted value:

| Tax Head | Dependent Variable | Stage-2 Structural Regressor |
|----------|--------------------|-------------------------------|
| **Income Tax (DT)** | `log_dt` | `log_gdp_hat`, `log_lsm_hat` |
| **GST** | `log_gst` | `log_gdp`, `log_imports`, `inflation` |
| **FED** | `log_fed` | `log_lsm_hat`, `log_gdp_hat` |
| **Customs** | `log_customs` | `log_dutiable_imports_hat`, `log_imports_hat` |

A `regime` dummy is included where structurally motivated (captures discrete shifts in tax administration 
efficacy across political/fiscal regimes).

**Three candidate Stage-2 estimators** compete for each tax head in an expanding-window backtest:
- **ARDL** — captures dynamic adjustment with AIC lag selection
- **ARIMAX** — Wold representation with ARIMA(1,1,0) structure
- **DynamicLag** — partial adjustment/Koyck specification

The tournament winner (lowest sMAPE over 8 rolling origins) is declared the **policy winner** 
and used for scenario-based projection.
""")

    with st.expander("📐 ARDL — AutoRegressive Distributed Lag (Pesaran et al., 2001)"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Pesaran, Shin & Smith (2001), *J. of Applied Econometrics*")
            st.info(
                "Bounds-testing approach to cointegration. Applicable to regressors that are "
                "I(0), I(1), or fractionally integrated — avoids pre-testing bias inherent in "
                "Engle-Granger or Johansen procedures."
            )
            st.markdown("**Key properties**")
            st.success(
                "Consistent under mixed integration orders · "
                "Efficient with T ≈ 30–80 · "
                "Delivers SR dynamics + LR multipliers in one step"
            )
        with c2:
            st.markdown("""
**Specification:**

The ARDL(p, q₁, …, qₖ) model in log-levels:

$$\\log T_t = c + \\sum_{i=1}^{p} \\rho_i \\log T_{t-i} + \\sum_{j=0}^{q} \\gamma_j \\log \\hat{X}_{t-j} + \\delta D_t + \\varepsilon_t$$

where:
- $T_t$ = tax revenue (log-transformed)
- $\\hat{X}_t$ = Stage-1 fitted value of the structural tax base (e.g. `log_imports_hat`)
- $D_t$ = regime dummy
- $p, q$ = AIC-selected lag orders (constrained to ≤ n/8)

**Long-Run Multiplier (LRM):**
$$\\theta = \\frac{\\sum_{j=0}^{q} \\gamma_j}{1 - \\sum_{i=1}^{p} \\rho_i}$$

Interpretation: A permanent 1% rise in the tax base leads to a $\\theta$% permanent change in revenue.

**Short-Run Coefficient:** $\\gamma_0$ — the contemporaneous elasticity within the fiscal year.

**ARDL-ECM Reparameterisation (for diagnostics):**
$$\\Delta \\log T_t = \\alpha(\\log T_{t-1} - \\theta \\log \\hat{X}_{t-1}) + \\text{SR terms} + \\varepsilon_t$$

$\\alpha < 0$ confirms error correction — revenue converges back to its structural level after a shock. 
The magnitude $|\\alpha|$ is the speed of adjustment per year.
""")

    with st.expander("📡 ARIMAX — ARIMA with Exogenous Regressors (Box & Jenkins, 1976)"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Box, Jenkins & Reinsel (1976); Hamilton (1994) *Ch. 4*")
            st.info(
                "Extensions of the Wold decomposition theorem to include deterministic exogenous inputs. "
                "Implemented here as SARIMAX(1,1,0) via statsmodels MLE — a fixed order chosen for "
                "parsimony given the ~30-observation annual sample."
            )
            st.markdown("**Key properties**")
            st.success(
                "Handles unit-root non-stationarity via differencing · "
                "MLE estimation under Gaussian innovations · "
                "Conditional on past revenues + exogenous path"
            )
        with c2:
            st.markdown("""
**Specification (ARMAX(1,1,0) in differences):**

$$\\Delta \\log T_t = \\mu + \\phi \\Delta \\log T_{t-1} + \\sum_k \\beta_k X_{kt} + \\varepsilon_t, \\quad \\varepsilon_t \\sim \\mathcal{N}(0, \\sigma^2)$$

where:
- $\\Delta \\log T_t = \\log T_t - \\log T_{t-1}$ — first difference removes I(1) stochastic trend
- $\\phi$ — AR(1) coefficient on lagged revenue growth (persistence)
- $X_{kt}$ — exogenous stage-1 regressors: `log_imports_hat`, `log_exrate`, `inflation`, `regime`
- Parameters estimated via **Kalman filter / MLE**

**Level forecast reconstruction:**

$$\\log \\hat{T}_{t+h} = \\log T_t + \\sum_{s=1}^{h} \\Delta \\log \\hat{T}_{t+s}$$

Recursive h-step forecasts accumulate first-difference predictions back to log-levels, 
then exponentiate. Exogenous paths are supplied from the scenario engine's channel projections.

**Note on order selection:** In the backtesting loop ARIMA(1,1,0) is used for computational 
efficiency (no auto-order search per fold). The final full-sample fit repeats this specification.
""")

    with st.expander("🔁 Dynamic Lag — Partial Adjustment / Koyck Model"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Koyck (1954); Nerlove (1958) partial adjustment model")
            st.info(
                "Reduces the infinite distributed lag to a single lagged dependent variable "
                "under geometric decay assumption. Equivalent to OLS on the Koyck-transformed "
                "equation; BLUE under standard Gauss-Markov conditions."
            )
            st.markdown("**Key properties**")
            st.success(
                "Parsimonious (2–3 free parameters) · "
                "OLS closed-form solution · "
                "Directly interpretable SR and LR elasticities"
            )
        with c2:
            st.markdown("""
**Partial Adjustment Specification:**

Assume desired (equilibrium) log-revenue $T^*_t = \\alpha + \\beta \\hat{X}_t + \\delta D_t$.  
Actual adjustment is partial: $\\log T_t - \\log T_{t-1} = \\lambda(\\log T^*_t - \\log T_{t-1})$

Substituting and rearranging yields the **estimable Koyck equation**:

$$\\log T_t = c + \\rho \\log T_{t-1} + \\beta^* \\log \\hat{X}_t + \\delta^* D_t + u_t$$

where $\\rho = 1 - \\lambda$ is the retention/persistence coefficient, and $\\beta^* = \\lambda \\beta$.

**Elasticity Recovery:**

| Horizon | Formula | Interpretation |
|---------|---------|----------------|
| Short-Run | $\\hat{\\beta}^*$ | Elasticity of revenue to base within the fiscal year |
| Long-Run | $\\hat{\\beta}^* / (1 - \\hat{\\rho})$ | Permanent elasticity after full convergence |

**Estimation:** OLS with heteroskedasticity-robust SEs (HC3). Note: Durbin's h-statistic 
(not DW) is the appropriate test for serial correlation given $\\log T_{t-1}$ on the RHS.

**Stage-2 input:** $\\hat{X}_t$ is the Stage-1 ARDL fitted value — using the structural estimate 
rather than the observed value eliminates the endogeneity-induced Nickell bias in the lagged-DV coefficient.
""")

    with st.expander("🧮 Elastic Net — Penalised Regression (Zou & Hastie, 2005)"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Zou & Hastie (2005), *J. of the Royal Statistical Society: Series B*")
            st.info(
                "A regularized regression method that linearly combines L1 (Lasso) and L2 (Ridge) penalties. "
                "It encourages sparsity (like Lasso) while also handling highly correlated predictors (like Ridge), "
                "making it suitable for high-dimensional data or when predictor selection is desired."
            )
            st.markdown("**Key properties**")
            st.success(
                "Automatic variable selection · "
                "Handles multicollinearity · "
                "Robust to overfitting with small N"
            )
        with c2:
            st.markdown("""
**Specification:**

The Elastic Net objective function minimizes:

$$\\min_{\\beta_0, \\beta} \\frac{1}{2n} \\sum_{i=1}^{n} (y_i - \\beta_0 - x_i^T \\beta)^2 + \\lambda \\left( (1-\\alpha) \\frac{1}{2} \\|\\beta\\|_2^2 + \\alpha \\|\\beta\\|_1 \\right)$$

where:
- $y_i$ = tax revenue (log-transformed)
- $x_i$ = vector of Stage-1 fitted values of structural tax bases (e.g., `log_gdp_hat`, `log_lsm_hat`) and regime dummies
- $\\lambda \\ge 0$ is the regularization parameter controlling the overall strength of the penalty.
- $\\alpha \\in [0, 1]$ is the mixing parameter between Lasso ($\\alpha=1$) and Ridge ($\\alpha=0$).

**Implementation:**
- `sklearn.linear_model.ElasticNet` is used.
- `alpha` (mixing parameter) is fixed at `0.5` (equal mix of L1 and L2).
- `l1_ratio` (equivalent to $\\alpha$) is set to `0.5`.
- `alpha` (regularization strength, equivalent to $\\lambda$) is tuned via cross-validation within the backtesting loop.
- Coefficients are estimated via coordinate descent.

**Advantages:**
- Can select groups of correlated variables.
- Provides a balance between prediction accuracy and interpretability.
- Effective in situations where the number of predictors is larger than the number of observations.
""")

# ─── TAB 7 — Data Preview ──────────────────────────────────────────────────
with tab7:
    st.markdown("""
    <div class="content-section">
        <div class="section-header">
            <div>
                <div class="section-title">Data Preview</div>
                <div class="section-subtitle">Historical data up to 2025 (read-only) · Budget Estimate (Current FY) for model context</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 1. Prepare Unified Dataset (History + Budget 2026)
    _df_to_show = df_hist if is_mm and df_hist is not None else df_raw
    _log_cols   = [c for c in _df_to_show.columns if c.startswith("log_")]
    _edit_cols  = [c for c in _df_to_show.columns if c not in _log_cols]
    _df_clean   = _df_to_show[_edit_cols].reset_index(drop=True)

    # Historical portion (≤2025)
    _hist_data = _df_clean[_df_clean["year_end"] <= 2025].copy()

    # Budget anchor (2026)
    if "revised_2026_df" not in st.session_state:
        _raw_full = load_tax_data()
        if _raw_full is not None and "year_end" in _raw_full.columns and 2026 in _raw_full["year_end"].values:
            _init_bud = _raw_full[_raw_full["year_end"] == 2026][_edit_cols].reset_index(drop=True)
        else:
            _init_bud = _df_clean.iloc[-1:].copy().reset_index(drop=True)
            if "year_end" in _init_bud.columns:
                _init_bud["year_end"] = 2026
        st.session_state["revised_2026_df"] = _init_bud

    # 2. Render Unified Data Preview (History + 2026 Row)
    # ── Persistence controls ──────────────────────────────────────────────────
    _persist_exists = _username and os.path.isfile(_persist_file(_username))
    _p_col1, _p_col2 = st.columns([5, 1])
    with _p_col1:
        if not _username:
            st.warning("⚠️ Enter a username in the sidebar to save your 2026 edits across sessions.")
        elif _persist_exists:
            st.success(f"✅ Saved data loaded for **{_username}**. Changes auto-save on edit.", icon=None)
        else:
            st.info(f"ℹ️ Logged in as **{_username}** — edit the 2026 row below to save.")
    with _p_col2:
        if _persist_exists and st.button("🔄 Reset to Defaults", help="Clear saved 2026 overrides and reload from source data"):
            try:
                os.remove(_persist_file(_username))
            except Exception:
                pass
            for _k in ("revised_2026_df", "editor_last_year", "user_df"):
                st.session_state.pop(_k, None)
            mm_get_cached_forecast.clear()
            st.rerun()

    # Shock Decomposition Components Preview (if shock mode is active)
    if use_demand_supply_shocks:
        with st.expander("📊 Inflation Shock Decomposition Components", expanded=True):
            st.markdown("#### Decomposed Inflation Components")
            
            shock_cols = ['demand_inflation_component', 'supply_inflation_component', 
                          'predicted_inflation', 'inflation_residual_shock', 'inflation']
            
            # Check which columns exist
            available_shock_cols = [col for col in shock_cols if col in df_raw.columns]
            
            if len(available_shock_cols) > 0:
                # Check if year_end is index or column
                if 'year_end' in df_raw.columns:
                    shock_display_df = df_raw[['year_end'] + available_shock_cols].copy()
                else:
                    shock_display_df = df_raw.reset_index()[['year_end'] + available_shock_cols].copy()
                
                # Display last 10 years
                st.dataframe(
                    shock_display_df.tail(10).style.format({
                        col: "{:.2f}" for col in shock_display_df.columns if col != 'year_end'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show decomposition statistics
                if 'inflation_decomp_coefs' in st.session_state:
                    coefs = st.session_state['inflation_decomp_coefs']
                    
                    st.markdown("**Inflation Decomposition Model Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R-squared", f"{coefs['r_squared']:.3f}")
                    with col2:
                        st.metric("Observations", f"{coefs['n_obs']}")
                    with col3:
                        if 'inflation_residual_shock' in df_raw.columns:
                            avg_residual = df_raw['inflation_residual_shock'].abs().mean()
                            st.metric("Avg |Residual|", f"{avg_residual:.2f}%")
                    
                    st.markdown("**Regression Coefficients:**")
                    coef_data = {
                        'Variable': ['Consumption Growth', 'LSM Growth', 'Imports Growth', 
                                     'Exchange Rate Growth', 'Oil Price Growth', 'Policy Rate'],
                        'Coefficient': [
                            coefs['consumption_growth'], coefs['lsm_growth'], 
                            coefs['imports_growth'], coefs['exrate_growth'],
                            coefs['oil_price_growth'], coefs['policy_rate']
                        ],
                        'Component': ['Demand', 'Demand', 'Demand', 'Supply', 'Supply', 'Supply']
                    }
                    coef_df = pd.DataFrame(coef_data)
                    st.dataframe(
                        coef_df.style.format({'Coefficient': '{:.4f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.info("""
                    **Interpretation:**
                    - **Demand Component**: Driven by consumption, industrial activity (LSM), and imports
                    - **Supply Component**: Driven by exchange rate changes, oil prices, and monetary policy
                    - **Residual Shock**: Unexplained inflation after accounting for demand and supply factors
                    """)
            else:
                st.info("Shock decomposition components will appear here when shock mode is enabled.")
        
        st.markdown("---")

    st.markdown("#### 🛠️ Model Training Data (History & 2026 Revised)")

    _unified_df = pd.concat([_hist_data, st.session_state["revised_2026_df"]], axis=0, ignore_index=True)

    _bud_col_cfg: dict = {}
    for col in _unified_df.columns:
        if col == "year_end":
            _bud_col_cfg[col] = st.column_config.NumberColumn("Year End", disabled=True, format="%d")
        elif col in ("covid", "regime", "step_2024", "dummy_2024", "dummy_2025",
                     "dummy_1995", "dummy_1996", "dummy_2002", "dummy_2003"):
            _bud_col_cfg[col] = st.column_config.CheckboxColumn(col)
        else:
            _bud_col_cfg[col] = st.column_config.NumberColumn(col, format="%.2f")

    edited_unified = st.data_editor(
        _unified_df,
        num_rows="fixed",
        use_container_width=True,
        column_config=_bud_col_cfg,
        key="unified_data_preview",
        hide_index=True
    )

    # Detect Change logic
    if not edited_unified.equals(_unified_df):
        # Extract the 2026 row from the edited unified dataframe
        _edited_bud_row = edited_unified[edited_unified["year_end"] == 2026].copy().reset_index(drop=True)
        _old_bud_row = st.session_state["revised_2026_df"].copy().reset_index(drop=True)

        if not _edited_bud_row.equals(_old_bud_row):
            # Budget changed -> Save and Retrain
            st.session_state["revised_2026_df"] = _edited_bud_row
            _save_2026_overrides(_username)
            # Build models dataset: History + NEW Budget 2026
            _combined_work = pd.concat([_hist_data, _edited_bud_row], axis=0, ignore_index=True)
            if "year_end" in _combined_work.columns:
                _combined_work["year_end"] = _combined_work["year_end"].astype(int)
                _combined_work.index = pd.PeriodIndex(_combined_work["year_end"], freq="Y") 
            _combined_work = prepare_transforms(_combined_work)
            st.session_state["user_df"] = _combined_work
            mm_get_cached_forecast.clear()
            st.session_state.pop("dyn_pipeline", None)
            st.rerun()
        else:
            # User tried to edit history -> Discard and refresh to original
            st.warning("Historical data (1995–2025) is read-only. Changes were discarded.")
            st.rerun()

    st.markdown("---")

    # 4. Budget Estimates (Current FY) — Editable · KPI Only
    st.markdown("#### 🎯 Comparison Benchmarks (2026 Budget Estimates)")

    if "editor_last_year" not in st.session_state:
        _init_re = _df_clean[_df_clean["year_end"] == 2026].copy().reset_index(drop=True)
        if _init_re.empty:
            _init_re = st.session_state["revised_2026_df"].copy().reset_index(drop=True)
        # Standard defaults for Revised Estimates comparison
        for c, v in [("dt", 6524000), ("gst", 4345000), ("fed", 898000), ("customs", 1312000)]:
            if c in _init_re.columns:
                _init_re[c] = float(v)
        st.session_state["editor_last_year"] = _init_re

    _re_col_cfg: dict = {}
    for col in st.session_state["editor_last_year"].columns:
        if col == "year_end":
            _re_col_cfg[col] = st.column_config.NumberColumn("Year End", disabled=True, format="%d")
        elif col in ("covid", "regime", "step_2024", "dummy_2024", "dummy_2025",
                     "dummy_1995", "dummy_1996", "dummy_2002", "dummy_2003"):
            _re_col_cfg[col] = st.column_config.CheckboxColumn(col)
        else:
            _re_col_cfg[col] = st.column_config.NumberColumn(col, format="%.2f")

    edited_re = st.data_editor(
        st.session_state["editor_last_year"],
        num_rows="fixed",
        use_container_width=True,
        column_config=_re_col_cfg,
        key="last_year_editor",
    )

    if not edited_re.equals(st.session_state["editor_last_year"]):
        st.session_state["editor_last_year"] = edited_re.copy()
        _save_2026_overrides(_username)
        st.rerun()

    st.markdown("#### Generated Future Exogenous Variables")
    if is_mm and exog_future is not None:
        st.dataframe(exog_future, use_container_width=True)
    elif not is_mm and dyn_results:
        st.write("Dynamic engine generates exog internally via channel equations.")
        st.json({
            k: (v if not isinstance(v, (list, pd.Series)) else list(v))
            for k, v in targets.items()
        })

    st.markdown("</div>", unsafe_allow_html=True)

# INSIGHTS PANEL
# ============================================================================
if fore_total is not None and len(fore_total) > 0 and fore_head is not None and len(fore_head) > 0:
    last_forecast_year = int(df_raw.index.max().year) + horizon if df_raw is not None else horizon
    st.markdown(f"""
    <div class="insight-panel">
        <div class="insight-header">
            <div class="insight-icon">💡</div>
            <div class="insight-title">Executive Insights (FY 2027)</div>
        </div>
        <div class="insight-content">
            The <strong>{default_model_label}</strong> model projects
            <strong>{TAX_LABELS[head]}</strong> revenue at <strong>Rs. {insight_cat_val:,.2f} Billion</strong>
            for FY 2027, representing a <strong>{insight_cat_growth:+.1f}%</strong> change
            from Revised Estimates (Current FY). Aggregate revenue across all tax heads for Next FY is expected to reach
            <strong>Rs. {total_fore_2027:,.2f} Billion</strong>, with a growth rate of
            <strong>{total_cagr_final:.2f}%</strong> relative to current revised estimates. The model achieved an out-of-sample error (MAE) of
            <strong>{mae_display}</strong> during validation.
        </div>
    </div>
    """, unsafe_allow_html=True)
# ─── TAB 8 — Sensitivity Analysis ──────────────────────────────────────────
with tab8:
    st.markdown("""
    <div class="content-section">
    <h2 class="section-title">🎛️ Sensitivity Analysis</h2>
    <p class="section-subtitle">Analyze revenue impact of oil price shocks</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Oil price elasticity (from econometric analysis)
    TOTAL_TAX_OIL_ELASTICITY = -0.21  # Updated: Log-log regression with lags
    
    st.markdown("### 🛢️ Supply Shock Sensitivity")
    
    # Methodology explanation
    with st.expander("📖 How was the elasticity estimated?"):
        st.markdown("""
        **Estimation Methodology: Three Econometric Approaches**
        
        The oil-price elasticity was estimated and validated using **three alternative econometric methods** 
        to ensure robustness and reliability of the estimate:
        
        ---
        
        ### **Method 1: Log-Log Regression with Lags** (Primary Method)
        
        ```
        log(Total Tax Revenue) = β₀ + β₁·log(Oil Price PKR) + β₂·log(Oil Price PKR)ₜ₋₁ 
                                + β₃·log(GDP) + β₄·Policy Rate + ε
        ```
        
        **Results:**
        - Contemporaneous elasticity: -0.2352 (p=0.016, significant at 5% level**)
        - Lagged elasticity: +0.0253 (not significant)
        - **Total elasticity: -0.2099**
        
        **Why this method?**
        - Log-log specification gives true elasticity (percentage interpretation)
        - PKR-adjusted oil prices (oil_price_brent × exrate) account for currency effects
        - Controls for GDP and policy rate to isolate oil effect
        - Includes lagged oil prices to capture delayed impacts
        
        ---
        
        ### **Method 2: Error Correction Model (ECM)** (Validation)
        
        Separates short-run and long-run effects:
        
        **Results:**
        - Short-run elasticity: +0.0635 (immediate adjustment)
        - **Long-run elasticity: -0.2391** (equilibrium effect)
        - Adjustment speed: -0.0843 (8.4% correction per year)
        - Half-life of shock: 7.9 years
        
        **Why this method?**
        - Distinguishes temporary vs permanent effects
        - Captures error correction dynamics
        - Confirms long-run negative relationship
        
        ---
        
        ### **Method 3: Rolling Window Analysis** (Time-Varying)
        
        Estimates elasticity over moving 10-year windows to detect structural changes:
        
        **Results:**
        - 2006-2015 period: **Positive** (+0.17 to +0.42) - oil boom era
        - 2016-2026 period: **Negative** (-0.02 to -0.23) - post-2014 oil crash
        - **Recent elasticity (2026): -0.1786**
        - Average (full sample): +0.0356
        
        **Why this method?**
        - Reveals that elasticity has changed over time
        - Recent estimates (2016-2026) show stronger negative relationship
        - Confirms structural shift in Pakistan's oil-revenue dynamics
        
        ---
        
        ### **Consensus Estimate: -0.21**
        
        All three methods independently confirm a **negative elasticity** around -0.21:
        
        | Method | Elasticity | Status |
        |--------|-----------|--------|
        | Log-log (total) | -0.210 | ✅ Statistically significant |
        | ECM (long-run) | -0.239 | ✅ Confirms negative relationship |
        | Rolling window (recent) | -0.179 | ✅ Shows recent trend |
        | **Recommended** | **-0.21** | ✅ **Robust across methods** |
        
        ---
        
        ### **Economic Interpretation**
        
        **Why is the elasticity negative?**
        
        Pakistan is an oil **importer** (~85% of consumption). When oil prices rise:
        
        1. **Import bill increases** → Drains foreign exchange reserves
        2. **Currency depreciates** → Amplifies oil shock in PKR terms (oil×exrate↑)
        3. **Inflation accelerates** → Real incomes and purchasing power fall
        4. **Economic activity slows** → GDP growth declines, manufacturing (LSM) contracts
        5. **Tax base shrinks** → GST, FED, Customs, and Direct Tax revenues all fall
        6. **Net effect**: Revenue decline dominates any gains from petroleum taxes
        
        ---
        
        ### **Methodological Advantages**
        
        The log-log specification with validation provides:
        - ✅ True elasticity measurement (percentage interpretation)
        - ✅ PKR-adjusted oil prices (captures currency amplification effects)
        - ✅ Control variables for GDP and policy rate (isolates oil effect)
        - ✅ Multiple independent validation methods
        - ✅ Economic consistency with Pakistan's oil-importer status
        - ✅ International best practice in elasticity estimation
        
        ---
        
        ### **Statistical Robustness**
        
        - **Sample period:** 1995-2026 (31 years of data)
        - **Significance:** p=0.016 (significant at 5% level)
        - **R-squared:** Model explains ~20% of tax revenue variation
        - **Validation:** Three methods converge on same sign and magnitude
        - **Economic consistency:** Negative elasticity is theoretically correct for oil importers
        
        ---
        
        ### **Economic Context**
        
        The negative elasticity is consistent with economic theory for oil-importing countries:
        - Oil importers experience GDP contraction when oil prices rise
        - Currency depreciation amplifies the oil shock in local currency terms
        - Net effect: Tax revenue falls despite higher petroleum taxes
        
        Pakistan's estimate of -0.21 indicates **moderate oil sensitivity** - the negative revenue 
        effect is partially offset by petroleum taxation and other revenue sources.
        """)
    
    st.markdown("""
    **Applied Elasticity:** -0.21 (PKR terms)
    
    **Interpretation:** A 10% oil price increase → approximately **2.1% decrease** in total tax revenue.
    
    **Validation:** Confirmed by three independent econometric methods (Log-log, ECM, Rolling window).
    """)
    
    # Oil price shock input - changed from slider to number input
    col_input1, col_input2 = st.columns([2, 1])
    
    with col_input1:
        oil_shock_pct = st.number_input(
            "Oil price shock (% change in PKR terms)",
            min_value=-100.0,
            max_value=200.0,
            value=0.0,
            step=5.0,
            format="%.1f",
            help="Enter the percentage change in oil prices (PKR terms). Positive = price increase, Negative = price decrease."
        )
    
    with col_input2:
        st.markdown("### Quick Scenarios")
        if st.button("🔺 +20% shock"):
            oil_shock_pct = 20.0
        if st.button("🔻 -30% shock"):
            oil_shock_pct = -30.0
        if st.button("↩️ Reset"):
            oil_shock_pct = 0.0
    
    # Get baseline total revenue (current forecast)
    if fore_total is not None and len(fore_total) > 0:
        # Use the first forecast year (typically next year)
        # Handle different formats of fore_total
        try:
            if isinstance(fore_total, pd.Series):
                baseline_total_revenue = float(fore_total.iloc[0])
            elif isinstance(fore_total, pd.DataFrame):
                # Get the first row, first numeric column
                baseline_total_revenue = float(fore_total.iloc[0, 0])
            else:
                # numpy array or similar
                baseline_total_revenue = float(fore_total[0])
        except:
            st.error("Could not extract baseline revenue. Please generate a forecast first.")
            baseline_total_revenue = 0
        
        # Calculate impact
        revenue_impact_pct = TOTAL_TAX_OIL_ELASTICITY * oil_shock_pct
        shock_adjusted_total_revenue = baseline_total_revenue * (1 + revenue_impact_pct / 100)
        revenue_difference = shock_adjusted_total_revenue - baseline_total_revenue
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Baseline Forecast",
                value=f"Rs. {baseline_total_revenue:,.0f}M",
                help="Total tax revenue forecast without oil shock"
            )
        
        with col2:
            st.metric(
                label="After Oil Shock",
                value=f"Rs. {shock_adjusted_total_revenue:,.0f}M",
                delta=f"{revenue_impact_pct:+.2f}%",
                help=f"Revenue after {oil_shock_pct:+.0f}% oil price shock"
            )
        
        with col3:
            st.metric(
                label="Revenue Difference",
                value=f"Rs. {revenue_difference:+,.0f}M",
                delta=f"{revenue_difference:+,.0f}M",
                help="Absolute revenue change"
            )
        
        # Explanation
        st.markdown("---")
        st.markdown("### 📊 Shock Impact Breakdown")
        
        impact_data = pd.DataFrame({
            "Metric": [
                "Oil price shock",
                "Tax elasticity",
                "Revenue impact",
                "Baseline revenue",
                "Adjusted revenue",
                "Revenue change"
            ],
            "Value": [
                f"{oil_shock_pct:+.1f}%",
                f"{TOTAL_TAX_OIL_ELASTICITY:.3f}",
                f"{revenue_impact_pct:+.2f}%",
                f"Rs. {baseline_total_revenue:,.0f}M",
                f"Rs. {shock_adjusted_total_revenue:,.0f}M",
                f"Rs. {revenue_difference:+,.0f}M"
            ]
        })
        
        st.dataframe(impact_data, use_container_width=True, hide_index=True)
        
        # Chart
        st.markdown("### 📈 Scenario Comparison")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Baseline',
            x=['Total Revenue'],
            y=[baseline_total_revenue],
            marker_color='#3B82F6'
        ))
        
        fig.add_trace(go.Bar(
            name=f'After {oil_shock_pct:+.0f}% Oil Shock',
            x=['Total Revenue'],
            y=[shock_adjusted_total_revenue],
            marker_color='#F59E0B' if oil_shock_pct > 0 else '#10B981'
        ))
        
        fig.update_layout(
            barmode='group',
            yaxis_title="Revenue (PKR Million)",
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Note
        st.info("""
        **Note:** This sensitivity analysis assumes:
        - Total tax oil-price elasticity: **-0.21** (log-log regression, 1995-2026)
        - Negative elasticity reflects Pakistan's oil-importer status
        - Linear approximation holds for moderate shocks (±30%)
        - All other macroeconomic factors remain constant (ceteris paribus)
        - Elasticity is in PKR terms (accounts for both USD oil price and exchange rate effects)
        - Based on contemporaneous effect; long-run impact may differ
        """)
    
    else:
        st.warning("⚠️ Please generate a forecast first to use sensitivity analysis.")
    
    st.markdown("</div>", unsafe_allow_html=True)
