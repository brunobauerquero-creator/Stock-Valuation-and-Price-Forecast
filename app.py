import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# --- USER CONFIGURATION ---
API_KEY = st.secrets["EODHD_API_KEY"]
HEADERS = {"Content-Type": "application/json"}
FRED_API = "https://api.stlouisfed.org/fred"
FRED_TOKEN = st.secrets["FRED_API_KEY"]

# --- MACRO DATA (US ONLY for now) ---
def fetch_us_macro():
    def get_fred_series(series_id):
        url = f"{FRED_API}/series/observations?series_id={series_id}&api_key={FRED_TOKEN}&file_type=json"
        r = requests.get(url).json()
        values = [float(o['value']) for o in r['observations'] if o['value'] != '.']
        return values[-1] if values else None

    return {
        "GDP_Growth": get_fred_series("A191RL1Q225SBEA"),
        "Unemployment": get_fred_series("UNRATE"),
        "Inflation": get_fred_series("CPIAUCSL"),
        "Interest_Rate": get_fred_series("FEDFUNDS")
    }

# --- FUNDAMENTALS ---
def get_fundamentals(ticker):
    url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={API_KEY}&fmt=json"
    return requests.get(url).json()

# --- INSIDER DATA ---
def calculate_insider_score(fundamental_json):
    insiders = fundamental_json.get("InsiderTransactions", [])
    score = 0
    for tx in insiders:
        tx_date = datetime.strptime(tx["filerDate"], "%Y-%m-%d")
        if datetime.now() - tx_date <= timedelta(days=180):
            mult = 2 if any(role in tx.get("filerName", "") for role in ["CEO", "CFO"]) else 1
            value = float(tx.get("value", 0))
            score += mult * (value if tx["type"] == "Buy" else -value)
    return score

# --- TECHNICAL SCORE (Simplified) ---
def get_technical_score(ticker):
    df = yf.download(ticker.split(".")[0], period="6mo")
    if df.empty: return 50
    rsi = df['Close'].pct_change().rolling(14).mean().iloc[-1] * 100
    return np.clip(50 + rsi, 0, 100)

# --- NORMALIZATION ---
def normalize(val, min_val, max_val, invert=False):
    if max_val - min_val == 0: return 50
    score = 100 * (val - min_val) / (max_val - min_val)
    return 100 - score if invert else score

# --- SCORE ENGINE ---
def compute_total_score(fund, macro, insider_val, tech_score):
    val_pe = float(fund.get("Valuation", {}).get("TrailingPE", 30))
    roe = float(fund.get("Highlights", {}).get("ReturnOnEquityTTM", 0))
    margin = float(fund.get("Highlights", {}).get("ProfitMargin", 0))

    f_score = (
        normalize(val_pe, 5, 60, invert=True)*0.4 +
        normalize(roe, 0, 50)*0.3 +
        normalize(margin, 0, 1)*0.3
    )

    m_score = (
        normalize(macro['GDP_Growth'], -5, 5)*0.4 +
        normalize(macro['Inflation'], 0, 10, invert=True)*0.3 +
        normalize(macro['Unemployment'], 2, 10, invert=True)*0.2 +
        normalize(macro['Interest_Rate'], 0, 6, invert=True)*0.1
    )

    i_score = normalize(insider_val, -1e6, 1e6)

    total = 0.4 * f_score + 0.2 * m_score + 0.2 * i_score + 0.2 * tech_score
    return total, f_score, m_score, i_score, tech_score

# --- STREAMLIT UI ---
st.set_page_config("Valuation Heatmap & Forecast")
st.title("📊 Stock Valuation Dashboard + Forecast & Insider Signals")

macro_data = fetch_us_macro()
st.sidebar.write("### US Macro Regime")
st.sidebar.json(macro_data)

st.subheader("1. Valuation Heatmap")
tickers = st.text_input("Enter tickers (comma-separated)", "AAPL.US,MSFT.US,TSLA.US,MCD.US,VTI.US").split(",")
data = []

for t in tickers:
    try:
        f = get_fundamentals(t)
        insider_val = calculate_insider_score(f)
        tech_score = get_technical_score(t)
        total, fs, ms, ins, tech = compute_total_score(f, macro_data, insider_val, tech_score)
        data.append({"Ticker": t.strip(), "Total": total, "Fundamental": fs, "Macro": ms, "Insider": ins, "Technical": tech})
    except:
        continue

df = pd.DataFrame(data)
if not df.empty:
    fig = px.imshow(df.set_index("Ticker")[["Total", "Fundamental", "Macro", "Insider", "Technical"]], color_continuous_scale="RdYlGn")
    st.plotly_chart(fig)

# --- FORWARD RETURN FORECAST ---
st.subheader("2. Forward Return Forecast (Simplified)")
tgt = st.selectbox("Select Ticker to Forecast", df["Ticker"] if not df.empty else [])

if tgt:
    yf_ticker = tgt.split(".")[0]
    df_hist = yf.download(yf_ticker, period="5y")
    df_hist['Future_3M'] = df_hist['Close'].pct_change(63).shift(-63)
    df_hist['Score'] = df_hist['Close'].pct_change(63).rolling(63).mean()
    st.line_chart(df_hist[['Close']].dropna())
    st.scatter_chart(df_hist[['Score', 'Future_3M']].dropna())

# --- INSIDER SIGNAL ---
st.subheader("3. Insider Buy Signal")
signal_df = df[df['Insider'] > 75]
st.write("**Strong Insider Buying Signals (last 6 months):**")
st.dataframe(signal_df.sort_values("Insider", ascending=False))
