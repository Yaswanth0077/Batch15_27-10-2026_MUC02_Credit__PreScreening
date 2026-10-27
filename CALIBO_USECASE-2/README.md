# KLU MUC02 — Credit Pre-Screening Engine

## Team

| Role | Member | Student ID |
|------|--------|------------|
| **Data Analyst** | Velagapudi Haritha | 2300030248 |
| **ML Engineer** | Kunapuli Bala Sri Sarvani | 2300030957 |
| **Project Lead & Advanced ML** | Yaswanth Kumar Reddy Vogireddy | 2300031126 |

**Programme:** Calibo AI Academy · Path 2 · KLU Batch  
**Assignment:** MUC02 — Loan Intake Optimisation via Credit Pre-Screening  

---

## Project Structure

```
KLU USECASE-2/
├── MUC02_Credit_PreScreening_Dataset 1.csv   ← 12,000 rows × 19 columns
├── YASWANTH_MUC02_KLU_Notebook.ipynb         ← Solution 1: Steps 5.1–5.7 (Better)
├── YASWANTH_MUC02_KLU_Advanced.ipynb         ← Solution 2: Advanced ML (Excellent)
├── YASWANTH_MUC02_KLU_Master.ipynb           ← Combined master notebook
├── app.py                                      ← Streamlit dashboard (6 tabs)
└── requirements.txt
```

## What Each Notebook Covers

### Solution 1 — Better (`YASWANTH_MUC02_KLU_Notebook.ipynb`)
- **Haritha:** Data cleaning — 3 planted issues fixed with full justification
- **Haritha:** EDA — credit score cliff at 650, employment type gap
- **Sarvani:** Feature engineering — FOIR, Loan-to-Income, is_ntc flag
- **Sarvani:** Logistic Regression pre-screener (Steps 5.4–5.6)
- **Yaswanth:** Credit policy briefing (Step 5.7)

### Solution 2 — Excellent (`YASWANTH_MUC02_KLU_Advanced.ipynb`)
- **Yaswanth:** 4 models: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Yaswanth:** Hyperparameter tuning (RandomizedSearchCV)
- **Yaswanth:** SHAP explainability (global, beeswarm, waterfall)
- **Yaswanth:** Two-lane pre-screener (auto-decline + auto-approve)

## Key Results

| Metric | Value |
|--------|-------|
| Dataset | 12,000 applications, 53% Approved |
| Best model | XGBoost (Tuned) ~85% accuracy, ~0.92 AUC |
| Auto-decline threshold | **0.90** |
| Apps auto-declined/month | ~456 (15.2%) |
| Analyst hours freed/month | ~190 hrs |
| Wrong-decline rate | **4.7%** (under 5% limit ✓) |

## Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. New App → select this repo → `app.py`
4. Deploy!
