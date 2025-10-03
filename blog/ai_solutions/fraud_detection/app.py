import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import altair as alt

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🕵️", layout="wide")
st.title("🕵️ Fraud Detection Demo (Streamlit + IsolationForest)")
st.write("Upload a CSV of transactions or use the sample to detect anomalous transactions.")

@st.cache_data
def load_sample():
    rng = np.random.RandomState(42)
    n = 500
    # Synthetic features: amount, time_delta, merchant_risk, user_txn_count
    amount = np.concatenate([rng.gamma(shape=2., scale=50., size=n-10), rng.gamma(10., 200., size=10)])
    time_delta = np.concatenate([rng.exponential(scale=60., size=n-10), rng.exponential(5., size=10)])
    merchant_risk = rng.uniform(0, 1, size=n)
    user_txn_count = rng.poisson(lam=10, size=n)
    df = pd.DataFrame({
        'transaction_id': np.arange(1, n+1),
        'amount': amount,
        'time_delta_sec': time_delta,
        'merchant_risk': merchant_risk,
        'user_txn_count_30d': user_txn_count,
    })
    return df

uploaded = st.file_uploader("Upload CSV (columns like: amount, time_delta_sec, merchant_risk, user_txn_count_30d)", type=["csv"]) 

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if st.button("Use sample data", type="primary"):
        st.session_state["use_sample"] = True
    if st.session_state.get("use_sample", False):
        df = load_sample()
    else:
        df = None

if df is None:
    st.info("Upload a CSV or click 'Use sample data' to proceed.")
    st.stop()

st.subheader("Input preview")
st.dataframe(df.head(20), use_container_width=True)

# Select features
default_features = [c for c in ["amount", "time_delta_sec", "merchant_risk", "user_txn_count_30d"] if c in df.columns]
features = st.multiselect("Select numeric feature columns for anomaly detection", options=list(df.columns), default=default_features)

if len(features) < 2:
    st.warning("Select at least two numeric feature columns.")
    st.stop()

X = df[features].select_dtypes(include=[np.number]).fillna(0.0).to_numpy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.header("Model settings")
contamination = st.sidebar.slider("Expected fraud rate (contamination)", 0.01, 0.2, 0.05, 0.01)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

model = IsolationForest(n_estimators=200, contamination=contamination, random_state=int(random_state))
model.fit(X_scaled)

scores = -model.score_samples(X_scaled)  # higher => more anomalous
preds = model.predict(X_scaled)  # -1 anomaly, 1 normal

result = df.copy()
result["anomaly_score"] = scores
result["is_fraud"] = (preds == -1)

st.subheader("Results")
st.write(f"Flagged {int(result['is_fraud'].sum())} potential frauds out of {len(result)} transactions.")
st.dataframe(result.sort_values("anomaly_score", ascending=False).head(50), use_container_width=True)

# Visualization: amount vs time with color by fraud if available
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) >= 2:
    x_col = st.selectbox("X-axis", options=num_cols, index=0)
    y_col = st.selectbox("Y-axis", options=num_cols, index=min(1, len(num_cols)-1))
    chart = alt.Chart(result).mark_circle(size=60).encode(
        x=alt.X(x_col, scale=alt.Scale(zero=False)),
        y=alt.Y(y_col, scale=alt.Scale(zero=False)),
        color=alt.condition(alt.datum.is_fraud, alt.value("red"), alt.value("steelblue")),
        tooltip=["transaction_id"] + num_cols[:6] + ["anomaly_score", "is_fraud"]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Add more numeric columns for scatter visualization.")

st.caption("This demo uses IsolationForest for basic anomaly detection. Not production guidance.")
