# Fraud Detection Demo (Streamlit + IsolationForest)

This directory contains a minimal working fraud detection demo using Streamlit with a simple anomaly detection model (IsolationForest) over financial transaction features. It supports CSV upload or a built-in synthetic sample and visualizes flagged anomalies.

## Quick start

- Python 3.9+
- Install deps (CPU-friendly):

```bash
pip install -U streamlit scikit-learn pandas numpy altair
streamlit run app.py
```

## Architecture

+----------------------+      +---------------------------+
|  Streamlit Frontend  | <--> |   Controller (Session)    |
+-----------+----------+      +---------------+-----------+
            |                                 |
            v                                 v
    File upload / sample                Feature selection, scaling
            |                                 |
            v                                 v
+----------------------+      +---------------------------+
| StandardScaler       | ---> | IsolationForest           |
| (normalize features) |      | (score + predict anomaly) |
+----------------------+      +---------------------------+
                                        |
                                        v
                              Results table + Altair chart

- Stateless inference: features are scaled each run; IF model is re-fit per interaction for simplicity.
- Replace with AutoEncoder or other models if desired (e.g., PyTorch/TF) for reconstruction error.

## Files

- app.py: Streamlit app with upload/sample, model fit, results table, and scatter plot.
- README.md: This file with setup, architecture, and references.

## Demo behavior

- Upload CSV with numeric columns (e.g., amount, time_delta_sec, merchant_risk, user_txn_count_30d) or click "Use sample data".
- Select feature columns used by the anomaly detector.
- Adjust contamination (expected fraud rate) in the sidebar, then view flagged transactions.
- Scatter plot colors suspected fraud in red.

## References

- Isolation Forest: Liu, Ting, Zhou (2008). https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf
- scikit-learn IsolationForest docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- Streamlit docs: https://docs.streamlit.io/
- Altair docs: https://altair-viz.github.io/

## Notes for Hugging Face Space integration

- Add a requirements.txt with pinned versions for Spaces (e.g., streamlit, scikit-learn, pandas, numpy, altair).
- Expose as a tab in a multi-demo Space launcher (e.g., Conversational AI, Fraud Detection).
- Consider lightweight models and CPU-only constraints on free Spaces tiers.
