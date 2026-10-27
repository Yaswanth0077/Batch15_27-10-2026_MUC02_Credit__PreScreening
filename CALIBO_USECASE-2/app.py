from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Pre-Screening Engine | KLU MUC02",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #0f1117; }
.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
    border: 1px solid #3a3f5c;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.metric-card h2 { color: #7c83fd; font-size: 2rem; font-weight: 700; margin: 0; }
.metric-card p  { color: #a0a8c0; font-size: 0.85rem; margin: 4px 0 0 0; }
.section-header {
    background: linear-gradient(90deg, #7c83fd22, transparent);
    border-left: 4px solid #7c83fd;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 12px 0;
}
.insight-box {
    background: #1a1f2e;
    border: 1px solid #2e3450;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
}
.approved-tag { background: #1a3a2a; color: #2ecc71; border-radius: 6px; padding: 2px 10px; font-size:0.8rem; }
.declined-tag { background: #3a1a1a; color: #e74c3c; border-radius: 6px; padding: 2px 10px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'approved': '#2ecc71', 'declined': '#e74c3c',
    'primary': '#7c83fd',  'secondary': '#f39c12',
    'bg': '#1e2130',       'card': '#252a3d',
    'text': '#e0e4f0',     'muted': '#7a8099'
}

# ─── DATA LOADING ──────────────────────────────────────────────────────────
@st.cache_data
def load_and_clean():
    BASE_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(BASE_DIR / "MUC02_Credit_PreScreening_Dataset 1.csv")
    df['monthly_income'] = df.groupby('employment_type')['monthly_income'].transform(
        lambda x: x.fillna(x.median()))
    df['is_ntc'] = (df['credit_score'].isna()).astype(int)
    df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
    df.loc[df['monthly_income'] > 600000, 'monthly_income'] /= 12
    df['foir'] = (df['existing_emi'] + df['proposed_emi']) / df['monthly_income']
    df['loan_to_income'] = df['loan_amount'] / (df['monthly_income'] * 12)
    df['target'] = (df['decision'] == 'Declined').astype(int)
    def score_band(row):
        if row['is_ntc'] == 1: return 'No Score'
        s = row['credit_score']
        if s < 600:   return '<600'
        elif s < 650: return '600–649'
        elif s < 700: return '650–699'
        elif s < 750: return '700–749'
        elif s < 800: return '750–799'
        else:         return '800+'
    df['score_band'] = df.apply(score_band, axis=1)
    def foir_band(f):
        if f < 0.30:  return '<30%'
        elif f < 0.40: return '30–40%'
        elif f < 0.50: return '40–50%'
        elif f < 0.55: return '50–55%'
        else:          return '>55%'
    df['foir_band'] = df['foir'].apply(foir_band)
    return df

df = load_and_clean()

# ─── TRAIN MODEL (cached) ─────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except: xgb_available = False
    try:
        from lightgbm import LGBMClassifier
        lgbm_available = True
    except: lgbm_available = False

    NUMERIC = ['credit_score', 'is_ntc', 'foir', 'loan_to_income', 'delinquencies_24m',
               'num_active_loans', 'years_in_current_job', 'age', 'tenure_months', 'city_tier']
    CATEGORICAL = ['employment_type', 'loan_purpose']
    X = df[NUMERIC + CATEGORICAL]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERIC),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL)
    ])

    def make_pipe(model):
        return Pipeline([('pre', preprocessor), ('model', model)])

    models_dict = {
        'Logistic Regression': make_pipe(LogisticRegression(max_iter=1000, random_state=42)),
        'Random Forest':       make_pipe(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    }
    if xgb_available:
        models_dict['XGBoost'] = make_pipe(XGBClassifier(n_estimators=100, random_state=42,
                                                           use_label_encoder=False, eval_metric='logloss', verbosity=0))
    if lgbm_available:
        models_dict['LightGBM'] = make_pipe(LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))

    results = {}
    for name, pipe in models_dict.items():
        pipe.fit(X_train, y_train)
        yp = pipe.predict(X_test)
        yprob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yprob)
        results[name] = {
            'pipeline': pipe,
            'accuracy': accuracy_score(y_test, yp) * 100,
            'f1':       f1_score(y_test, yp) * 100,
            'auc':      roc_auc_score(y_test, yprob) * 100,
            'cm':       confusion_matrix(y_test, yp),
            'fpr': fpr, 'tpr': tpr,
            'y_prob': yprob, 'y_test': y_test
        }
    return results, X_test, y_test, preprocessor, NUMERIC, CATEGORICAL

results, X_test, y_test, preprocessor, NUMERIC, CATEGORICAL = train_models(df)
best_model_name = max(results, key=lambda k: results[k]['auc'])
best_pipe = results[best_model_name]['pipeline']

# ─── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Credit Pre-Screening")
    st.markdown("**KLU MUC02 · Calibo AI Academy**")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Overview",
        "📊 EDA & Insights",
        "🤖 Model Comparison",
        "🎯 Live Prediction",
        "📋 Credit Policy Briefing"
    ])
    st.markdown("---")
    st.markdown(f"**Dataset:** 12,000 applications")
    st.markdown(f"**Approval Rate:** {(df['decision']=='Approved').mean()*100:.1f}%")
    st.markdown(f"**Best Model:** {best_model_name}")
    st.markdown(f"**Best AUC:** {results[best_model_name]['auc']:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🏦 Credit Pre-Screening Engine")
    st.markdown("### Loan Intake Optimisation · KLU Batch · Calibo AI Academy")
    st.markdown("---")

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("12,000", "Total Applications"),
        (f"{(df['decision']=='Approved').mean()*100:.1f}%", "Approval Rate"),
        (f"{results[best_model_name]['auc']:.1f}%", "Best Model AUC"),
        ("0.90", "Recommended Threshold"),
        ("~190 hrs", "Analyst Hours Freed/Month"),
    ]
    for col, (val, label) in zip([col1,col2,col3,col4,col5], metrics):
        with col:
            st.markdown(f"""<div class="metric-card"><h2>{val}</h2><p>{label}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("### 📖 Business Context")
        st.markdown("""
A mid-size private bank receives **~3,000 personal loan applications per month**.
Every application goes through a **25-minute manual credit review**.

Nearly **half are declined** — many for reasons visible in the first minute:
- Credit score below 650
- FOIR above 55%

This pre-screening model lets analysts **focus on cases that actually need human judgement**.
        """)
        st.markdown("""
| Threshold | Auto-Declined | Wrong Declines | Hours Freed/Month |
|---|---|---|---|
| 0.50 | ~44.9% | ~20.9% ❌ | Too risky |
| 0.80 | ~24.1% | ~5.2% ⚠️ | ~361 hrs |
| **0.90** | **~15.2%** | **~4.7% ✅** | **~190 hrs** |
| 0.95 | ~7.3% | ~2.1% ✅ | ~91 hrs |
        """)
    with col_r:
        # Decision distribution donut
        dec_counts = df['decision'].value_counts()
        fig = px.pie(values=dec_counts.values, names=dec_counts.index,
                     color=dec_counts.index,
                     color_discrete_map={'Approved': COLORS['approved'], 'Declined': COLORS['declined']},
                     hole=0.55, title="Overall Decision Distribution")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color=COLORS['text'], legend=dict(orientation='h', y=-0.05))
        fig.update_traces(textinfo='percent+label', textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA & Insights":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.markdown("Uncovering the bank's unwritten credit policy.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Credit Score", "FOIR & Affordability", "Employment & Demographics", "Correlations"])

    with tab1:
        band_order = ['<600', '600–649', '650–699', '700–749', '750–799', '800+', 'No Score']
        score_approval = (df.groupby('score_band')['decision']
                          .apply(lambda x: (x=='Approved').mean()*100)
                          .reindex(band_order).reset_index())
        score_approval.columns = ['Band', 'Approval Rate (%)']
        fig = px.bar(score_approval, x='Band', y='Approval Rate (%)',
                     color='Approval Rate (%)',
                     color_continuous_scale=['#e74c3c','#f39c12','#2ecc71'],
                     title='Approval Rate by Credit Score Band',
                     text=score_approval['Approval Rate (%)'].round(1).astype(str)+'%')
        fig.add_hline(y=5, line_dash='dash', line_color='red', annotation_text='~5% cliff',
                      annotation_position='top right')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color=COLORS['text'], coloraxis_showscale=False, height=400)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.info("🔍 **The 650 Cliff:** Approval jumps from **4.3%** to **32.8%** at the 650 score line — an unwritten bank policy the pre-screener can encode with confidence.")
        st.info("🔍 **NTC Applicants:** Approved **53.9%** of the time — the credit team uses human judgement here, not just scores.")

    with tab2:
        foir_order = ['<30%', '30–40%', '40–50%', '50–55%', '>55%']
        foir_approval = (df.groupby('foir_band')['decision']
                         .apply(lambda x: (x=='Approved').mean()*100)
                         .reindex(foir_order).reset_index())
        foir_approval.columns = ['FOIR Band', 'Approval Rate (%)']
        fig = px.bar(foir_approval, x='FOIR Band', y='Approval Rate (%)',
                     color='Approval Rate (%)',
                     color_continuous_scale=['#e74c3c','#f39c12','#2ecc71'],
                     title='Approval Rate by FOIR Band (Fixed Obligations to Income Ratio)',
                     text=foir_approval['Approval Rate (%)'].round(1).astype(str)+'%')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color=COLORS['text'], coloraxis_showscale=False, height=400)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        st.info("🔍 Approval falls from **65.4%** (FOIR <30%) to **15.5%** (FOIR >55%). Sharpest drop beyond 50% — suggest policy FOIR limit of **50–55%**.")

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.histogram(df[df['foir']<2], x='foir', color='decision',
                                color_discrete_map={'Approved':COLORS['approved'],'Declined':COLORS['declined']},
                                barmode='overlay', nbins=60, opacity=0.7,
                                title='FOIR Distribution by Decision')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color=COLORS['text'], height=300)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.box(df[df['monthly_income']<300000], x='decision', y='monthly_income',
                          color='decision',
                          color_discrete_map={'Approved':COLORS['approved'],'Declined':COLORS['declined']},
                          title='Monthly Income by Decision')
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color=COLORS['text'], height=300,
                               yaxis_title='Monthly Income (₹)')
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            emp_approval = (df.groupby('employment_type')['decision']
                           .apply(lambda x: (x=='Approved').mean()*100).reset_index())
            emp_approval.columns = ['Employment Type', 'Approval Rate (%)']
            emp_approval = emp_approval.sort_values('Approval Rate (%)')
            fig = px.bar(emp_approval, y='Employment Type', x='Approval Rate (%)',
                         orientation='h', color='Approval Rate (%)',
                         color_continuous_scale=['#e74c3c','#2ecc71'],
                         title='Approval Rate by Employment Type',
                         text=emp_approval['Approval Rate (%)'].round(1).astype(str)+'%')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color=COLORS['text'], coloraxis_showscale=False, height=300)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            gender_approval = (df.groupby('gender')['decision']
                              .apply(lambda x: (x=='Approved').mean()*100).reset_index())
            gender_approval.columns = ['Gender', 'Approval Rate (%)']
            fig2 = px.bar(gender_approval, x='Gender', y='Approval Rate (%)',
                          color='Gender', color_discrete_sequence=[COLORS['declined'], COLORS['primary']],
                          title='Approval Rate by Gender',
                          text=gender_approval['Approval Rate (%)'].round(1).astype(str)+'%')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color=COLORS['text'], showlegend=False, height=300)
            fig2.update_traces(textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)

        # Monthly volume heatmap
        df['month'] = pd.to_datetime(df['application_date']).dt.to_period('M').astype(str)
        monthly = df.groupby(['month','decision']).size().unstack(fill_value=0).reset_index()
        fig3 = go.Figure()
        fig3.add_bar(x=monthly['month'], y=monthly['Approved'], name='Approved',
                     marker_color=COLORS['approved'])
        fig3.add_bar(x=monthly['month'], y=monthly['Declined'], name='Declined',
                     marker_color=COLORS['declined'])
        fig3.update_layout(barmode='stack', title='Monthly Application Volume',
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           font_color=COLORS['text'], height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        numeric_cols = ['age', 'monthly_income', 'existing_emi', 'credit_score',
                        'num_active_loans', 'delinquencies_24m', 'loan_amount',
                        'proposed_emi', 'foir', 'loan_to_income', 'target']
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, color_continuous_scale='RdBu_r', aspect='auto',
                        title='Feature Correlation Heatmap', text_auto='.2f',
                        zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=COLORS['text'], height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Target = 1 means Declined. Blue = negative correlation (higher → more likely approved). Red = positive (higher → more likely declined).")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    st.markdown("# 🤖 Model Comparison")
    st.markdown("Comparing all 4 trained models side-by-side.")
    st.markdown("---")

    # Metrics table
    metrics_df = pd.DataFrame({
        name: {'Accuracy (%)': f"{r['accuracy']:.1f}", 'F1 Score (%)': f"{r['f1']:.1f}", 'ROC-AUC (%)': f"{r['auc']:.1f}"}
        for name, r in results.items()
    }).T
    st.markdown("### 📊 Performance Metrics")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color='#2a3a2a'), use_container_width=True)

    # ROC Curves
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        colors_roc = ['#3498db','#2ecc71','#e74c3c','#9b59b6']
        for i, (name, r) in enumerate(results.items()):
            fig.add_trace(go.Scatter(x=r['fpr'], y=r['tpr'], mode='lines',
                                     name=f"{name} ({r['auc']:.1f}%)",
                                     line=dict(color=colors_roc[i%len(colors_roc)], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random',
                                  line=dict(color='grey', dash='dash'), showlegend=False))
        fig.update_layout(title='ROC Curves — All Models',
                          xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font_color=COLORS['text'], height=420,
                          legend=dict(x=0.5, y=0.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sel_model = st.selectbox("View Confusion Matrix for:", list(results.keys()))
        cm = results[sel_model]['cm']
        labels = ['Approved', 'Declined']
        fig_cm = px.imshow(cm, x=['Pred: Approved','Pred: Declined'],
                           y=['Act: Approved','Act: Declined'],
                           color_continuous_scale='RdYlGn_r',
                           text_auto=True, aspect='auto',
                           title=f'Confusion Matrix — {sel_model}')
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=COLORS['text'], height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Threshold analysis for best model
    st.markdown("### 🎯 Threshold Analysis")
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    y_prob_best = results[best_model_name]['y_prob']
    thresh_rows = []
    for t in thresholds:
        mask = y_prob_best >= t
        n = mask.sum()
        pct_ad = n / len(y_test) * 100
        if n > 0:
            wrong = ((mask) & (y_test == 0)).sum()
            pct_wd = wrong / n * 100
        else:
            pct_wd = 0.0
        thresh_rows.append({'Threshold': t, 'Auto-Declined (%)': round(pct_ad,1), 'Wrong Declines (%)': round(pct_wd,1)})
    thresh_df = pd.DataFrame(thresh_rows)

    fig_t = make_subplots(specs=[[{"secondary_y": True}]])
    fig_t.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Auto-Declined (%)'],
                                mode='lines+markers', name='Auto-Declined (%)',
                                line=dict(color='#3498db', width=2.5), marker_size=8), secondary_y=False)
    fig_t.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Wrong Declines (%)'],
                                mode='lines+markers', name='Wrong Declines (%)',
                                line=dict(color='#e74c3c', width=2.5, dash='dash'), marker_size=8), secondary_y=True)
    fig_t.add_hline(y=5, line_dash='dot', line_color='#e74c3c', secondary_y=True,
                    annotation_text='5% Risk Limit', annotation_position='top right')
    fig_t.add_vline(x=0.90, line_dash='dash', line_color='#7c83fd',
                    annotation_text='Chosen: 0.90', annotation_position='top left')
    fig_t.update_yaxes(title_text='Auto-Declined (%)', secondary_y=False)
    fig_t.update_yaxes(title_text='Wrong Declines (%)', secondary_y=True)
    fig_t.update_layout(title=f'Threshold Analysis — {best_model_name}',
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         font_color=COLORS['text'], height=380)
    st.plotly_chart(fig_t, use_container_width=True)
    st.success("✅ **Recommended Threshold: 0.90** — frees ~190 analyst hours/month with <5% wrong-decline rate")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Live Prediction":
    st.markdown("# 🎯 Live Applicant Prediction")
    st.markdown("Enter applicant details to get an instant pre-screening decision.")
    st.markdown("---")

    col_form, col_result = st.columns([1.1, 1])

    with col_form:
        st.markdown("### Applicant Details")
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 21, 60, 32)
                monthly_income = st.number_input("Monthly Income (₹)", 12000, 500000, 55000, step=1000)
                credit_score = st.slider("Credit Score (0 if NTC)", 0, 890, 720)
                existing_emi = st.number_input("Existing EMI (₹/month)", 0, 100000, 4000, step=500)
                proposed_emi = st.number_input("Proposed EMI (₹/month)", 1000, 200000, 8000, step=500)
            with c2:
                employment_type = st.selectbox("Employment Type", df['employment_type'].unique())
                loan_purpose = st.selectbox("Loan Purpose", df['loan_purpose'].unique())
                loan_amount = st.number_input("Loan Amount (₹)", 50000, 2500000, 300000, step=10000)
                tenure_months = st.selectbox("Tenure (months)", [12, 24, 36, 48, 60], index=2)
                delinquencies_24m = st.slider("Late Payments (last 24m)", 0, 11, 0)
                num_active_loans = st.slider("Active Loans", 0, 8, 1)
                years_in_job = st.number_input("Years in Current Job", 0.0, 30.0, 3.0, step=0.5)
                city_tier = st.selectbox("City Tier", [1, 2, 3])
            submitted = st.form_submit_button("🔍 Predict Now", use_container_width=True)

    with col_result:
        st.markdown("### Prediction Result")
        if submitted:
            is_ntc = 1 if credit_score == 0 else 0
            cs = df['credit_score'].median() if is_ntc else credit_score
            foir = (existing_emi + proposed_emi) / monthly_income
            lti = loan_amount / (monthly_income * 12)

            input_df = pd.DataFrame([{
                'credit_score': cs, 'is_ntc': is_ntc, 'foir': foir, 'loan_to_income': lti,
                'delinquencies_24m': delinquencies_24m, 'num_active_loans': num_active_loans,
                'years_in_current_job': years_in_job, 'age': age, 'tenure_months': tenure_months,
                'city_tier': city_tier, 'employment_type': employment_type, 'loan_purpose': loan_purpose
            }])

            prob_decline = best_pipe.predict_proba(input_df)[0][1]
            prob_approve = 1 - prob_decline

            # Result card
            if prob_decline >= 0.90:
                decision = "🔴 AUTO-DECLINED"
                decision_color = COLORS['declined']
                note = "Model confidence ≥ 90%. Application flagged for auto-decline."
            elif prob_approve >= 0.85:
                decision = "🟢 AUTO-APPROVE (Fast Track)"
                decision_color = COLORS['approved']
                note = "Model confidence ≥ 85% for approval. Prioritise for quick review."
            elif prob_decline >= 0.50:
                decision = "🟡 SEND TO ANALYST"
                decision_color = COLORS['secondary']
                note = "Borderline case. Requires manual credit review."
            else:
                decision = "🟢 LIKELY APPROVED"
                decision_color = COLORS['approved']
                note = "Model confidence suggests approval. Verify with analyst."

            st.markdown(f"""<div class="metric-card" style="border-color:{decision_color};">
                <h2 style="color:{decision_color}; font-size:1.4rem;">{decision}</h2>
                <p>{note}</p></div>""", unsafe_allow_html=True)

            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_decline * 100,
                title={'text': "Decline Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': decision_color},
                    'steps': [
                        {'range': [0, 50],  'color': '#1a3a2a'},
                        {'range': [50, 85], 'color': '#3a3a1a'},
                        {'range': [85, 100],'color': '#3a1a1a'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=COLORS['text'], height=280)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Key signals
            st.markdown("**Key Signals:**")
            signals = {
                f"FOIR: {foir:.2f} ({foir*100:.0f}%)": "🔴" if foir > 0.55 else "🟡" if foir > 0.40 else "🟢",
                f"Credit Score: {credit_score}": "🔴" if credit_score < 600 else "🟡" if credit_score < 700 else "🟢",
                f"Delinquencies: {delinquencies_24m}": "🔴" if delinquencies_24m > 1 else "🟡" if delinquencies_24m == 1 else "🟢",
                f"NTC Applicant: {'Yes' if is_ntc else 'No'}": "🟡" if is_ntc else "🟢",
            }
            for label, icon in signals.items():
                st.markdown(f"{icon} {label}")
        else:
            st.markdown("""<div class="insight-box">
            <p style="color:#7a8099;">Fill in the applicant details and click <b>Predict Now</b> to get an instant pre-screening result.</p>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: CREDIT POLICY BRIEFING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📋 Credit Policy Briefing":
    st.markdown("# 📋 Credit Policy Briefing")
    st.markdown("*What I would say to the Credit Policy Analyst before she meets the Head of Retail Lending.*")
    st.markdown("---")

    st.markdown("""
> **"I recommend auto-declining a loan application only when the model is at least 90% confident it will be declined.**
> On our test data, that removes approximately **15.2% of applications — roughly 456 applications a month, or 190 analyst hours per month** — and fewer than 5% of those would have been approved by the credit team, staying within the bank's stated risk limit.

> **Two factors drive decisions most strongly:**
> - **FOIR** — approval rates fall from 65.4% (below 30%) to just 15.5% (above 55%). Any applicant above a 55% FOIR is declined over 8 times out of 10.
> - **Credit Score** — approvals are effectively zero below 650, and jump sharply at that mark. The model confirmed what the data shows: the bank has an unwritten score floor.

> **The model should not auto-decide for New-To-Credit (NTC) applicants.** It is correct only 56.7% of the time for them — barely better than a coin toss — because it has no bureau history to work from. All NTC applications should go directly to a human analyst.

> **On fairness:** Gender was not used as a feature, and the model's auto-decline rate is almost identical — 24.1% for female vs 23.1% for male applicants. However, income and employment type can act as proxy variables for gender, so I recommend reporting outcomes by gender every month and investigating any drift.

> **Before going live**, I would want **repayment data**: today the model is trained on *past analyst decisions*, not on whether customers actually repaid — and those two things are not always the same."
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class="metric-card">
        <h2>0.90</h2><p>Recommended Threshold</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
        <h2>~456</h2><p>Applications/month auto-declined</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
        <h2>~190 hrs</h2><p>Analyst time freed per month</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### The Two Strongest Drivers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="insight-box">
        <b style="color:#7c83fd;">FOIR (Fixed Obligation to Income Ratio)</b><br>
        Coefficient: +1.22 (strongest driver)<br>
        Approval: 65.4% below 30% → 15.5% above 55%<br>
        Proposed policy limit: <b>50–55% FOIR</b>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="insight-box">
        <b style="color:#7c83fd;">Credit Score</b><br>
        Coefficient: −1.01 (strongest approval signal)<br>
        Below 650: approval rate only 4.3%<br>
        The <b>650 threshold</b> is an unwritten bank policy
        </div>""", unsafe_allow_html=True)
