# British Airways — Customer Review Insights & Buying Behavior Modeling

Analyze British Airways customer reviews to extract insights, build a predictive model for buying behavior, and run business simulations to estimate impact on revenue and satisfaction.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make prepare
make train
make simulate
```
Artifacts will appear in `models/reports/` (metrics, simulations) and the trained model in `models/artifacts/`.

> Learning project using public/synthetic data; not affiliated with British Airways.

## Interactive Dashboard
Run locally:
```bash
streamlit run app.py
