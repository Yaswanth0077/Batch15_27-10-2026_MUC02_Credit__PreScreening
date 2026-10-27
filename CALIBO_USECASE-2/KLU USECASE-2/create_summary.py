"""
Creates Batch15_MUC02_KLU_Summary.docx for team submission.
Run: python create_summary.py
"""
try:
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

if HAS_DOCX:
    doc = Document()

    # Title
    title = doc.add_heading('MUC02 — Credit Pre-Screening Engine', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph('Calibo AI Academy · Path 2 · KLU Batch 15 | Team Submission')
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Team table
    doc.add_heading('Team', level=1)
    table = doc.add_table(rows=4, cols=3)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = 'Role', 'Member', 'Student ID'
    rows_data = [
        ('Data Analyst', 'Velagapudi Haritha', '2300030248'),
        ('ML Engineer', 'Kunapuli Bala Sri Sarvani', '2300030957'),
        ('Project Lead & Advanced ML', 'Yaswanth Kumar Reddy Vogireddy', '2300031126'),
    ]
    for i, (role, name, sid) in enumerate(rows_data):
        row = table.rows[i+1].cells
        row[0].text, row[1].text, row[2].text = role, name, sid

    # Problem Summary
    doc.add_heading('Problem Statement', level=1)
    doc.add_paragraph(
        'A mid-size private bank receives approximately 3,000 personal loan applications per month. '
        'Every application undergoes a 25-minute manual review. Nearly half are ultimately declined, '
        'many for reasons visible in the first minute. We were asked to build a logistic regression '
        'pre-screener that auto-declines applications with high confidence, freeing analyst time '
        'while keeping wrong-decline rate within 5%.'
    )

    # What we did
    doc.add_heading('What the Team Delivered', level=1)
    doc.add_paragraph(
        'Haritha identified and fixed three planted data quality issues: 441 missing monthly incomes '
        '(filled using employment-type medians), 1,039 blank credit scores belonging to New-To-Credit '
        '(NTC) applicants (flagged with is_ntc = 1), and 12 annual incomes entered as monthly figures '
        '(divided by 12). She then performed exploratory analysis revealing the unwritten credit score '
        'floor at 650 and the approval gap between Gig/Contract and Salaried applicants.'
    )
    doc.add_paragraph(
        'Sarvani built three affordability features (FOIR, Loan-to-Income ratio, NTC flag) and trained '
        'the logistic regression pre-screener. The model achieves 79% accuracy with a ROC-AUC of 0.87. '
        'A threshold analysis across seven cut-offs identified 0.90 as the optimal threshold — '
        'auto-declining 15.2% of applications (456/month, 190 analyst hours freed) with only 4.7% '
        'wrong-decline rate, within the bank\'s 5% risk limit. A fairness check confirmed the model '
        'auto-declines male and female applicants at nearly equal rates (24.1% F vs 23.1% M).'
    )
    doc.add_paragraph(
        'Yaswanth extended the solution to distinction level, training four models (Logistic Regression, '
        'Random Forest, XGBoost, LightGBM) with hyperparameter tuning on XGBoost (best CV-AUC ~0.92). '
        'SHAP global importance, beeswarm, and waterfall plots explain individual predictions. '
        'A two-lane pre-screener adds an auto-approve fast-track lane (P(Approve) ≥ 0.85) in addition '
        'to the auto-decline lane, automating a larger share of decisions with controlled risk. '
        'An interactive 6-tab Streamlit dashboard enables live predictions and SHAP explanations.'
    )

    # Key Results
    doc.add_heading('Key Results', level=1)
    rtable = doc.add_table(rows=5, cols=2)
    rtable.style = 'Table Grid'
    for label, val in [
        ('Chosen threshold', '0.90'),
        ('Applications auto-declined/month', '~456 (15.2%)'),
        ('Analyst hours freed/month', '~190 hrs'),
        ('Wrong-decline rate', '4.7% (under 5% limit)'),
        ('Best model AUC (XGBoost tuned)', '~0.92'),
    ]:
        i = [('Chosen threshold', '0.90'),
             ('Applications auto-declined/month', '~456 (15.2%)'),
             ('Analyst hours freed/month', '~190 hrs'),
             ('Wrong-decline rate', '4.7% (under 5% limit)'),
             ('Best model AUC (XGBoost tuned)', '~0.92')].index((label, val))
        row = rtable.rows[i].cells
        row[0].text, row[1].text = label, val

    # Policy position
    doc.add_heading('Credit Policy Recommendation', level=1)
    doc.add_paragraph(
        'We recommend auto-declining applications only when the model is at least 90% confident of '
        'decline. At this threshold, ~456 applications per month are auto-declined, freeing ~190 '
        'analyst hours, with a wrong-decline rate of 4.7%. FOIR and credit score are the two '
        'strongest drivers. NTC applicants should always go to a human analyst — the model is only '
        '56.7% accurate for them. Gender is excluded from the model; auto-decline rates are nearly '
        'equal by gender, but income and employment type should be monitored as potential proxies. '
        'Before live deployment, repayment data is needed: the model currently learns from analyst '
        'decisions, not whether customers actually repaid.'
    )

    out = r'e:\CaliboAiAcad\MCU2\CALIBO_USECASE-2\KLU USECASE-2\Batch15_MUC02_KLU_Summary.docx'
    doc.save(out)
    print(f'Summary saved: {out}')
else:
    # Fallback: write as plain text
    out = r'e:\CaliboAiAcad\MCU2\CALIBO_USECASE-2\KLU USECASE-2\Batch15_MUC02_KLU_Summary.txt'
    content = """MUC02 — Credit Pre-Screening Engine
Calibo AI Academy · Path 2 · KLU Batch 15 | Team Submission

TEAM
Data Analyst:            Velagapudi Haritha (2300030248)
ML Engineer:             Kunapuli Bala Sri Sarvani (2300030957)
Project Lead & Adv ML:   Yaswanth Kumar Reddy Vogireddy (2300031126)

KEY RESULTS
- Chosen threshold: 0.90
- Applications auto-declined/month: ~456 (15.2%)
- Analyst hours freed/month: ~190 hrs
- Wrong-decline rate: 4.7% (under 5% limit)
- Best model AUC (XGBoost tuned): ~0.92

RECOMMENDATION
Auto-decline at 90% model confidence. FOIR and credit score are top drivers.
NTC applicants → always human review. Monitor fairness by gender monthly.
"""
    with open(out, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Summary (txt fallback) saved: {out}')
    print('Install python-docx for proper .docx: pip install python-docx')
