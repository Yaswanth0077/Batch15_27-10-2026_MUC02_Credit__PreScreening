# MUC02 — Credit Pre-Screening Engine
## Team Summary | Batch 15 | Group Submission

**Calibo AI Academy · Path 2 · KLU Batch**

---

## Team

| Role | Member | Student ID |
|------|--------|------------|
| **Data Analyst** | Velagapudi Haritha | 2300030248 |
| **ML Engineer** | Kunapuli Bala Sri Sarvani | 2300030957 |
| **Project Lead & Advanced ML** | Yaswanth Kumar Reddy Vogireddy | 2300031126 |

---

## Problem Statement

A mid-size private bank receives approximately 3,000 personal loan applications per month. Every application undergoes a 25-minute manual review, and nearly half are ultimately declined — many for reasons visible within the first minute. The Head of Retail Lending asked: can a model pre-screen applications so analysts spend their time where judgement is genuinely needed?

---

## What the Team Delivered

**Haritha (Data Analyst)** identified and fixed three planted data quality issues: 441 missing monthly incomes filled using employment-type medians, 1,039 blank credit scores belonging to New-To-Credit (NTC) applicants flagged with `is_ntc = 1`, and 12 annual incomes incorrectly entered as monthly figures divided by 12. She then performed exploratory analysis, revealing the unwritten credit score floor at 650 (approval jumps from 4.3% to 32.8% across this line) and the significant approval gap between Gig/Contract workers (34.9%) and Salaried applicants (58.4%).

**Sarvani (ML Engineer)** engineered three affordability features — FOIR, Loan-to-Income ratio, and the NTC flag — and trained a Logistic Regression pre-screener (79% accuracy, ROC-AUC 0.87). A threshold analysis across seven cut-off values identified **0.90 as the optimal threshold**: auto-declining 15.2% of applications (~456/month, freeing ~190 analyst hours) with only a 4.7% wrong-decline rate — within the bank's 5% risk limit. A fairness check confirmed nearly equal auto-decline rates by gender (24.1% F vs 23.1% M).

**Yaswanth (Project Lead & Advanced ML)** extended the work to distinction level: trained and compared four models (LR, Random Forest, XGBoost, LightGBM) with hyperparameter tuning on XGBoost achieving ~0.92 CV-AUC. SHAP global importance, beeswarm, and waterfall plots provide full explainability. A two-lane pre-screener adds an auto-approve fast-track lane. An interactive 6-tab Streamlit dashboard enables live predictions and SHAP explanations.

---

## Key Results

| Metric | Value |
|--------|-------|
| Chosen threshold | **0.90** |
| Applications auto-declined/month | **~456 (15.2%)** |
| Analyst hours freed/month | **~190 hrs** |
| Wrong-decline rate | **4.7%** (under 5% limit ✓) |
| Best model AUC (XGBoost tuned) | **~0.92** |

---

## Credit Policy Recommendation

We recommend auto-declining applications only when the model is **at least 90% confident of decline**. At this threshold, ~456 applications per month are auto-declined, freeing ~190 analyst hours, with a wrong-decline rate of 4.7% — within the bank's 5% risk limit.

FOIR and credit score are the two strongest drivers. **NTC applicants should always go to a human analyst** — the model is only 56.7% accurate for them. Gender is excluded from the model; auto-decline rates are nearly equal by gender, but income and employment type should be monitored as potential proxies. Before live deployment, repayment data is needed: the model currently learns from past analyst decisions, not whether customers actually repaid.

---

*Word count: ~490 words (within 500-word group limit)*
