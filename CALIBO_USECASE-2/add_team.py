import json, uuid

TEAM_HEADER = """# MUC02 — Credit Pre-Screening Engine
## Calibo AI Academy · Path 2 · KLU Batch

| Role | Member | Student ID |
|------|--------|------------|
| **Data Analyst** | Velagapudi Haritha | 2300030248 |
| **ML Engineer** | Kunapuli Bala Sri Sarvani | 2300030957 |
| **Project Lead & Advanced ML** | Yaswanth Kumar Reddy Vogireddy | 2300031126 |

**Role Responsibilities**
- **Haritha** → Data cleaning (Issues 1–3), EDA, credit score & employment analysis (Steps 5.1–5.2)
- **Sarvani** → Feature engineering, Logistic Regression, threshold analysis, fairness (Steps 5.3–5.6)
- **Yaswanth** → Policy briefing, 4-model comparison, SHAP, two-lane pre-screener, Streamlit dashboard (Step 5.7 + Solution 2)
"""

header_cell = {
    "cell_type": "markdown",
    "id": str(uuid.uuid4())[:8],
    "metadata": {},
    "source": [line + "\n" for line in TEAM_HEADER.strip().split("\n")]
}

notebooks = [
    "YASWANTH_MUC02_KLU_Notebook.ipynb",
    "YASWANTH_MUC02_KLU_Advanced.ipynb",
    "YASWANTH_MUC02_KLU_Master.ipynb"
]

for nb_file in notebooks:
    with open(nb_file, encoding='utf-8') as f:
        nb = json.load(f)
    # Replace first cell if already a team header, else prepend
    if nb['cells'] and 'Calibo AI Academy' in ''.join(nb['cells'][0].get('source',[])):
        nb['cells'][0] = {**header_cell, 'id': str(uuid.uuid4())[:8]}
    else:
        nb['cells'] = [{**header_cell, 'id': str(uuid.uuid4())[:8]}] + nb['cells']
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Updated: {nb_file} ({len(nb['cells'])} cells)")
