import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1e3a8a; margin-bottom: 0.2rem; }
    .subtitle { color: #64748b; font-size: 1.05rem; margin-bottom: 1.5rem; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD DATASET (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "HR_Attrition_dataset.csv"),
        os.path.join(base_dir, "HR_Attrition_dataset(3).csv"),
        "HR_Attrition_dataset.csv",
        "HR_Attrition_dataset(3).csv",
        "../HR_Attrition_dataset.csv",
        "c:/Users/jetly/Downloads/HR_Attrition_dataset.csv"
    ]
    data = None
    for p in candidates:
        if os.path.exists(p):
            data = pd.read_csv(p)
            break
    if data is None:
        st.error("HR_Attrition_dataset.csv could not be found. Please ensure the CSV is in the repository.")
        st.stop()

    data = data.drop_duplicates().copy()
    if "Attrition" in data.columns:
        data["Attrition"] = data["Attrition"].astype(str).str.strip().str.title()
    
    # Standardize target labels
    data = data.loc[data["Attrition"].isin(["Yes", "No"])].copy().reset_index(drop=True)
    return data

DATA = load_data()
ID_COL = "Employee_ID" if "Employee_ID" in DATA.columns else None
TARGET = "Attrition"

MODEL_FEATURES = [
    c for c in DATA.columns
    if c not in [TARGET, ID_COL] and DATA[c].nunique(dropna=True) > 1
]
NUM_COLS = DATA[MODEL_FEATURES].select_dtypes(include=np.number).columns.tolist()
CAT_COLS = [c for c in MODEL_FEATURES if c not in NUM_COLS]

def make_preprocessor(features):
    num = [c for c in features if c in NUM_COLS]
    cat = [c for c in features if c in CAT_COLS]
    transformers = []
    if num:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat))
    return ColumnTransformer(transformers, remainder="drop")

# -----------------------------------------------------------------------------
# TRAIN CLASSIFIERS (CACHED)
# -----------------------------------------------------------------------------
@st.cache_resource
def train_classifiers():
    X = DATA[MODEL_FEATURES].copy()
    y = DATA[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    definitions = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_leaf=10,
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    }
    models, rows, predictions = {}, [], {}
    for name, estimator in definitions.items():
        model = Pipeline([
            ("prep", make_preprocessor(MODEL_FEATURES)),
            ("classifier", estimator)
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        classes = list(model.classes_)
        prob = model.predict_proba(X_test)[:, classes.index("Yes")] if "Yes" in classes else np.zeros(len(y_test))
        y_binary = (y_test == "Yes").astype(int)
        try:
            auc = roc_auc_score(y_binary, prob)
        except Exception:
            auc = np.nan
        rows.append({
            "Algorithm": name,
            "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
            "Precision - Leave (%)": round(precision_score(y_test, pred, pos_label="Yes", zero_division=0) * 100, 2),
            "Recall - Leave (%)": round(recall_score(y_test, pred, pos_label="Yes", zero_division=0) * 100, 2),
            "F1 - Leave (%)": round(f1_score(y_test, pred, pos_label="Yes", zero_division=0) * 100, 2),
            "ROC-AUC": round(float(auc), 3) if not np.isnan(auc) else np.nan
        })
        models[name] = model
        predictions[name] = {
            "y_test": y_test, "pred": pred, "prob": prob,
            "confusion": confusion_matrix(y_test, pred, labels=["No", "Yes"])
        }
    return models, pd.DataFrame(rows), predictions

MODELS, COMPARISON_DF, MODEL_EVAL = train_classifiers()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/commercial-development-management.png", width=64)
    st.title("HR Intelligence")
    st.markdown("**Employee Attrition Analytics**")
    st.caption("Data Warehousing & Data Mining Project")
    st.markdown("---")
    selected_tab = st.radio(
        "Navigation",
        [
            "01 · Overview",
            "02 · K-Means Clustering",
            "03 · Classification Comparison",
            "04 · Association Rule Mining",
            "05 · Attrition Risk Prediction"
        ],
        index=0
    )
    st.markdown("---")
    st.info("💡 Powered by Streamlit Community Cloud and Scikit-Learn.")

# Header
st.markdown('<div class="main-title">Employee Attrition Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explore workforce patterns, clustering segmentation, model evaluations, and individual attrition risk scores.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -----------------------------------------------------------------------------
if selected_tab == "01 · Overview":
    st.subheader("Workforce Overview & Key Metrics")
    
    total = len(DATA)
    yes = int((DATA[TARGET] == "Yes").sum())
    no = int((DATA[TARGET] == "No").sum())
    rate = 100 * yes / total if total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Left Organization", f"{yes:,}", delta=f"{rate:.1f}% Attrition", delta_color="inverse")
    col3.metric("Retained Employees", f"{no:,}", delta=f"{100 - rate:.1f}% Retained")
    col4.metric("Attrition Rate", f"{rate:.2f}%")

    st.markdown("---")
    c1, c2 = st.columns([1, 1.2])

    with c1:
        counts = DATA[TARGET].value_counts().reindex(["No", "Yes"], fill_value=0).rename_axis("Status").reset_index(name="Employees")
        fig_pie = px.pie(
            counts, names="Status", values="Employees", hole=0.55,
            title="Attrition Distribution", color="Status",
            color_discrete_map={"No": "#16a34a", "Yes": "#ef4444"}
        )
        fig_pie.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("### Model Benchmark Comparison")
        st.dataframe(COMPARISON_DF.set_index("Algorithm"), use_container_width=True)
        st.caption("Models trained with balanced class weights on an 80/20 stratified split.")

    with st.expander("🔍 View Raw Dataset Preview"):
        st.dataframe(DATA.head(100), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: K-MEANS CLUSTERING
# -----------------------------------------------------------------------------
elif selected_tab == "02 · K-Means Clustering":
    st.subheader("Employee Clustering via K-Means")
    
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        selected_features = st.multiselect(
            "Select Features for Clustering:",
            options=MODEL_FEATURES,
            default=MODEL_FEATURES[:8]
        )
    with colB:
        k_val = st.slider("Number of Clusters (K):", min_value=2, max_value=8, value=3)
    with colC:
        sample_size = st.slider("PCA Visualization Sample:", min_value=200, max_value=len(DATA), value=min(1000, len(DATA)))

    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
    else:
        with st.spinner("Calculating clusters and PCA projection..."):
            Xraw = DATA[selected_features].copy()
            numeric = Xraw.select_dtypes(include=np.number).columns.tolist()
            categorical = [c for c in selected_features if c not in numeric]
            parts = []
            if numeric:
                parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric))
            if categorical:
                parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("one", OneHotEncoder(handle_unknown="ignore"))]), categorical))
            
            prep = ColumnTransformer(parts)
            X = prep.fit_transform(Xraw)
            X_dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            
            km = KMeans(n_clusters=int(k_val), random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels, sample_size=min(1500, len(labels)), random_state=42)

            # KPIs
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clusters (K)", int(k_val))
            m2.metric("Silhouette Score", f"{sil:.4f}")
            m3.metric("Inertia", f"{km.inertia_:,.1f}")
            m4.metric("Features Included", len(selected_features))

            st.markdown("---")
            g1, g2 = st.columns(2)
            with g1:
                counts = pd.Series(labels).value_counts().sort_index()
                cluster_df = pd.DataFrame({
                    "Cluster": [f"Cluster {i}" for i in range(int(k_val))],
                    "Employees": [int(counts.get(i, 0)) for i in range(int(k_val))]
                })
                fig_bar = px.bar(cluster_df, x="Cluster", y="Employees", color="Cluster", title=f"Employee Count by Cluster (K={k_val})")
                fig_bar.update_layout(template="plotly_white")
                st.plotly_chart(fig_bar, use_container_width=True)

            with g2:
                n_sample = min(int(sample_size), len(X_dense))
                rng = np.random.RandomState(42)
                idx = np.sort(rng.choice(len(X_dense), size=n_sample, replace=False)) if n_sample < len(X_dense) else np.arange(len(X_dense))
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_dense[idx])
                plot_df = pd.DataFrame({
                    "PC1": coords[:, 0], "PC2": coords[:, 1],
                    "Cluster": [f"Cluster {labels[i]}" for i in idx]
                })
                if ID_COL:
                    plot_df["Employee_ID"] = DATA.iloc[idx][ID_COL].astype(str).values
                fig_pca = px.scatter(
                    plot_df, x="PC1", y="PC2", color="Cluster",
                    hover_data=["Employee_ID"] if ID_COL else None,
                    title="Employee Clusters in 2D PCA Space",
                    opacity=0.75
                )
                fig_pca.update_layout(template="plotly_white")
                st.plotly_chart(fig_pca, use_container_width=True)

            st.markdown("### Cluster Attrition Breakdown")
            assigned = pd.DataFrame({"Cluster": labels, "Attrition": DATA[TARGET].values})
            profile = pd.crosstab(assigned["Cluster"], assigned["Attrition"], normalize="index").mul(100).round(2).reset_index()
            st.dataframe(profile, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: CLASSIFICATION COMPARISON
# -----------------------------------------------------------------------------
elif selected_tab == "03 · Classification Comparison":
    st.subheader("Decision Tree vs. Random Forest Comparison")

    metric_choice = st.selectbox(
        "Select Metric to Compare:",
        ["Accuracy (%)", "Precision - Leave (%)", "Recall - Leave (%)", "F1 - Leave (%)", "ROC-AUC"]
    )

    fig_comp = px.bar(
        COMPARISON_DF, x="Algorithm", y=metric_choice, color="Algorithm",
        title=f"Algorithm Performance on Held-Out Test Set ({metric_choice})"
    )
    fig_comp.update_layout(template="plotly_white", height=380)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Confusion Matrices")
    c1, c2 = st.columns(2)
    for idx, (name, col) in enumerate(zip(["Decision Tree", "Random Forest"], [c1, c2])):
        with col:
            cm = MODEL_EVAL[name]["confusion"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=["Predicted Stay", "Predicted Leave"],
                y=["Actual Stay", "Actual Leave"],
                color_continuous_scale="Blues",
                title=f"{name}: Confusion Matrix"
            )
            fig_cm.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4: ASSOCIATION RULE MINING
# -----------------------------------------------------------------------------
elif selected_tab == "04 · Association Rule Mining":
    st.subheader("Workforce Association Rule Mining (Apriori)")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        min_supp = st.slider("Min Support:", 0.05, 0.50, 0.10, 0.01)
    with col2:
        min_conf = st.slider("Min Confidence:", 0.20, 1.00, 0.50, 0.05)
    with col3:
        max_len = st.selectbox("Max Itemset Length:", [2, 3, 4], index=1)
    with col4:
        rank_by = st.selectbox("Rank Rules By:", ["Lift", "Confidence (%)", "Support (%)"], index=0)

    rule_cols = st.multiselect(
        "Select Categorical Columns for Mining:",
        options=CAT_COLS + [TARGET],
        default=[TARGET, "Overtime", "Job_Role", "Business_Travel"] if "Job_Role" in CAT_COLS else CAT_COLS[:4] + [TARGET]
    )

    if len(rule_cols) < 2:
        st.warning("Please select at least 2 columns to generate association rules.")
    else:
        with st.spinner("Extracting frequent itemsets and rules..."):
            transaction_df = DATA[rule_cols].copy()
            transactions = [
                [f"{col}={('Unknown' if pd.isna(row[col]) else str(row[col]).strip())}" for col in rule_cols]
                for _, row in transaction_df.iterrows()
            ]
            te = TransactionEncoder()
            arr = te.fit(transactions).transform(transactions)
            basket = pd.DataFrame(arr, columns=te.columns_)
            itemsets = apriori(basket, min_support=float(min_supp), use_colnames=True, max_len=int(max_len))

            if itemsets.empty:
                st.info("No frequent itemsets found. Try lowering the Support threshold.")
            else:
                rules = association_rules(itemsets, metric="confidence", min_threshold=float(min_conf))
                if rules.empty:
                    st.info("No rules met the selected confidence threshold.")
                else:
                    rules["Antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
                    rules["Consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
                    rules["Support (%)"] = (rules["support"] * 100).round(2)
                    rules["Confidence (%)"] = (rules["confidence"] * 100).round(2)
                    rules["Lift"] = rules["lift"].round(3)
                    
                    sort_key = "lift" if rank_by == "Lift" else ("confidence" if "Confidence" in rank_by else "support")
                    rules = rules.sort_values(sort_key, ascending=False)
                    out_df = rules[["Antecedents", "Consequents", "Support (%)", "Confidence (%)", "Lift"]]
                    
                    st.success(f"Generated {len(rules):,} association rules from {len(itemsets):,} frequent itemsets.")
                    
                    fig_rules = px.scatter(
                        out_df, x="Support (%)", y="Confidence (%)",
                        color="Lift", size="Lift", hover_data=["Antecedents", "Consequents"],
                        title="Association Rules: Support vs. Confidence",
                        color_continuous_scale="Viridis"
                    )
                    fig_rules.update_layout(template="plotly_white", height=450)
                    st.plotly_chart(fig_rules, use_container_width=True)
                    st.dataframe(out_df.head(50), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5: PREDICTION
# -----------------------------------------------------------------------------
elif selected_tab == "05 · Attrition Risk Prediction":
    st.subheader("Predict Individual Employee Attrition Risk")
    
    colM1, colM2 = st.columns(2)
    with colM1:
        chosen_model = st.selectbox("Select Prediction Model:", ["Random Forest", "Decision Tree"], index=0)
    with colM2:
        risk_threshold = st.slider("Risk Alert Threshold:", 0.10, 0.90, 0.50, 0.05)

    st.markdown("#### Employee Profile Attributes")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        p_age = st.number_input("Age", min_value=18, max_value=65, value=35)
        p_dept = st.selectbox("Department", sorted(DATA["Department"].dropna().unique()) if "Department" in DATA else ["Sales", "Research & Development", "HR"])
        p_salary = st.number_input("Salary / Monthly Rate", min_value=1000, max_value=200000, value=5000)
        p_overtime = st.selectbox("Overtime", ["Yes", "No"], index=1)
        p_years_company = st.number_input("Years at Company", 0, 40, 5)

    with r2:
        p_gender = st.selectbox("Gender", sorted(DATA["Gender"].dropna().unique()) if "Gender" in DATA else ["Male", "Female"])
        p_role = st.selectbox("Job Role", sorted(DATA["Job_Role"].dropna().unique()) if "Job_Role" in DATA else ["Sales Executive", "Research Scientist", "Laboratory Technician"])
        p_hike = st.slider("Salary Hike (%)", 0, 35, 14)
        p_travel = st.selectbox("Business Travel", sorted(DATA["Business_Travel"].dropna().unique()) if "Business_Travel" in DATA else ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        p_years_role = st.number_input("Years in Current Role", 0, 25, 3)

    with r3:
        p_distance = st.slider("Distance From Home (km)", 1, 60, 10)
        p_job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
        p_experience = st.number_input("Total Experience (Years)", 0, 45, 10)
        p_marital = st.selectbox("Marital Status", sorted(DATA["Marital_Status"].dropna().unique()) if "Marital_Status" in DATA else ["Single", "Married", "Divorced"])
        p_promotion = st.number_input("Years Since Promotion", 0, 20, 1)

    with r4:
        p_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        p_env = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        p_involvement = st.slider("Job Involvement (1-4)", 1, 4, 3)
        p_worklife = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        p_companies = st.slider("Prior Companies", 0, 10, 2)
        p_education = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=2)

    if st.button("🚀 Calculate Attrition Risk", type="primary", use_container_width=True):
        values = {
            "Age": p_age, "Business_Travel": p_travel, "Department": p_dept,
            "Distance_From_Home": p_distance, "Education": p_education,
            "Environment_Satisfaction": p_env, "Gender": p_gender,
            "Salary": p_salary, "Job_Involvement": p_involvement, "Job_Level": str(p_job_level),
            "Job_Role": p_role, "Job_Satisfaction": p_satisfaction,
            "Marital_Status": p_marital,
            "Number_of_Companies_Worked_previously": p_companies,
            "Overtime": p_overtime, "Salary_Hike_in_percent": p_hike,
            "Total_working_years_experience": p_experience,
            "Work_life_balance": p_worklife,
            "No_of_years_worked_at_current_company": p_years_company,
            "No_of_years_in_current_role": p_years_role,
            "Years_since_last_promotion": p_promotion
        }
        row = {}
        for col in MODEL_FEATURES:
            val = values.get(col, DATA[col].mode(dropna=True).iloc[0] if not DATA[col].mode(dropna=True).empty else 0)
            row[col] = val
        input_df = pd.DataFrame([row], columns=MODEL_FEATURES)

        model = MODELS[chosen_model]
        prob = float(model.predict_proba(input_df)[0, list(model.classes_).index("Yes")])
        is_high = prob >= risk_threshold

        st.markdown("---")
        res1, res2 = st.columns([1, 1.2])

        with res1:
            if is_high:
                st.error(f"⚠️ **HIGH ATTRITION RISK DETECTED**\n\nProbability of leaving: **{prob * 100:.1f}%** (Threshold: {risk_threshold * 100:.0f}%)")
            else:
                st.success(f"✅ **LOW ATTRITION RISK (RETAINED)**\n\nProbability of leaving: **{prob * 100:.1f}%** (Threshold: {risk_threshold * 100:.0f}%)")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 32}},
                title={'text': "Attrition Probability", 'font': {'size': 18}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if is_high else "#16a34a"},
                    'steps': [
                        {'range': [0, 30], 'color': "#dcfce7"},
                        {'range': [30, 60], 'color': "#fef9c3"},
                        {'range': [60, 100], 'color': "#fee2e2"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 3},
                        'thickness': 0.75,
                        'value': risk_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res2:
            st.markdown("### Profile Feature Snapshot")
            st.dataframe(pd.DataFrame([row]).T.rename(columns={0: "Value"}), use_container_width=True)
