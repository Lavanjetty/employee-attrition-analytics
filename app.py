# ============================================================
# DWDM EMPLOYEE ATTRITION ANALYTICS DASHBOARD
# Google Colab + Gradio
# Modules: Overview, K-Means, Classification Comparison,
# Association Rule Mining, Attrition Prediction
# ============================================================


import os, io, warnings
import numpy as np
import pandas as pd
import gradio as gr
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
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")

# ---- Load data: upload CSV in Colab if not already present ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_CANDIDATES = [
    os.path.join(BASE_DIR, "HR_Attrition_dataset.csv"),
    os.path.join(BASE_DIR, "HR_Attrition_dataset(3).csv"),
    "/content/HR_Attrition_dataset.csv",
    "/content/HR_Attrition_dataset(3).csv",
    "HR_Attrition_dataset.csv",
    "HR_Attrition_dataset(3).csv",
]

DATA = None
for path in CSV_CANDIDATES:
    if os.path.exists(path):
        DATA = pd.read_csv(path)
        print(f"Loaded dataset: {path}")
        break

if DATA is None:
    raise FileNotFoundError(
        "HR_Attrition_dataset.csv was not found. Upload this CSV file to the "
        "same repository/folder as app.py before launching the dashboard."
    )

DATA = DATA.drop_duplicates().copy()
if "Attrition" in DATA.columns:
    DATA["Attrition"] = DATA["Attrition"].astype(str).str.strip().str.title()

ID_COL = "Employee_ID" if "Employee_ID" in DATA.columns else None
TARGET = "Attrition"
if TARGET not in DATA.columns:
    raise ValueError("The dataset must contain an Attrition column.")

# Standardize target labels when dataset uses Yes/No
valid_target = DATA[TARGET].isin(["Yes", "No"])
DATA = DATA.loc[valid_target].copy().reset_index(drop=True)
DATA[TARGET] = DATA[TARGET].astype(str)

# Feature list for modeling; never use target/employee ID as predictors
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

def nice_fig(fig, height=420):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=25, r=25, t=65, b=35),
        font=dict(family="Arial", size=12),
        title_x=0.02
    )
    return fig

# Cache trained classification models for prediction and comparison
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
            "Accuracy (%)": round(accuracy_score(y_test, pred)*100, 2),
            "Precision - Leave (%)": round(precision_score(y_test, pred, pos_label="Yes", zero_division=0)*100, 2),
            "Recall - Leave (%)": round(recall_score(y_test, pred, pos_label="Yes", zero_division=0)*100, 2),
            "F1 - Leave (%)": round(f1_score(y_test, pred, pos_label="Yes", zero_division=0)*100, 2),
            "ROC-AUC": round(float(auc), 3) if not np.isnan(auc) else np.nan
        })
        models[name] = model
        predictions[name] = {
            "y_test": y_test, "pred": pred, "prob": prob,
            "confusion": confusion_matrix(y_test, pred, labels=["No", "Yes"])
        }
    return models, pd.DataFrame(rows), predictions

print("Training Decision Tree and Random Forest models for dashboard...")
MODELS, COMPARISON, MODEL_EVAL = train_classifiers()
print("Dashboard models ready.")

# ---- Overview ----
def overview():
    total = len(DATA)
    yes = int((DATA[TARGET] == "Yes").sum())
    no = int((DATA[TARGET] == "No").sum())
    rate = 100 * yes / total if total else 0
    counts = DATA[TARGET].value_counts().reindex(["No", "Yes"], fill_value=0).rename_axis("Status").reset_index(name="Employees")
    fig = px.pie(counts, names="Status", values="Employees", hole=0.55,
                 title="Attrition distribution", color="Status",
                 color_discrete_map={"No":"#16a34a", "Yes":"#ef4444"})
    fig = nice_fig(fig, 350)
    cards = f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Employees</div><div class="kpi-value">{total:,}</div></div>
      <div class="kpi"><div class="kpi-label">Left the organization</div><div class="kpi-value">{yes:,}</div></div>
      <div class="kpi"><div class="kpi-label">Retained employees</div><div class="kpi-value">{no:,}</div></div>
      <div class="kpi"><div class="kpi-label">Attrition rate</div><div class="kpi-value">{rate:.2f}%</div></div>
    </div>
    """
    return cards, fig, COMPARISON

# ---- K-Means ----
CLUSTER_FEATURES = [c for c in MODEL_FEATURES if c not in []]

def run_clustering(k, selected_features, compare_k, sample_size):
    features = selected_features or CLUSTER_FEATURES
    if len(features) < 2:
        raise gr.Error("Select at least two features for clustering.")
    if len(DATA) <= int(k):
        raise gr.Error("K must be smaller than the number of employees.")

    Xraw = DATA[features].copy()
    numeric = Xraw.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    parts = []
    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ])
        parts.append(("num", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        parts.append(("cat", cat_pipe, categorical))
    prep = ColumnTransformer(parts)
    X = prep.fit_transform(Xraw)
    if hasattr(X, "toarray"):
        X_dense = X.toarray()
    else:
        X_dense = np.asarray(X)
    km = KMeans(n_clusters=int(k), random_state=42, n_init=10)
    labels = km.fit_predict(X)
    counts = pd.Series(labels).value_counts().sort_index()
    cluster_df = pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in range(int(k))],
        "Employees": [int(counts.get(i, 0)) for i in range(int(k))]
    })
    fig_sizes = px.bar(cluster_df, x="Cluster", y="Employees",
                       color="Cluster", title=f"Employee count by cluster (K={int(k)})")
    fig_sizes = nice_fig(fig_sizes, 360)

    # PCA is only for visualizing the already fitted cluster assignments
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
        title="Employee clusters in 2D PCA space",
        opacity=0.75
    )
    fig_pca = nice_fig(fig_pca, 480)

    # Silhouette calculation can be expensive; sample rows for large datasets.
    if len(np.unique(labels)) > 1:
        try:
            sil = silhouette_score(
                X, labels,
                sample_size=min(2000, len(labels)),
                random_state=42
            )
        except Exception:
            sil = np.nan
    else:
        sil = np.nan

    metrics = f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-label">Clusters (K)</div><div class="kpi-value">{int(k)}</div></div>
      <div class="kpi"><div class="kpi-label">Silhouette score</div><div class="kpi-value">{sil:.4f}</div></div>
      <div class="kpi"><div class="kpi-label">Inertia</div><div class="kpi-value">{km.inertia_:,.1f}</div></div>
      <div class="kpi"><div class="kpi-label">Features used</div><div class="kpi-value">{len(features)}</div></div>
    </div>
    """
    assigned = DATA[[ID_COL]].copy() if ID_COL else pd.DataFrame(index=DATA.index)
    assigned["Cluster"] = labels
    if TARGET in DATA.columns:
        assigned["Attrition"] = DATA[TARGET].values
    # Profile / attrition by cluster
    if TARGET in DATA.columns:
        profile = pd.crosstab(assigned["Cluster"], assigned["Attrition"], normalize="index").mul(100).round(2).reset_index()
    else:
        profile = cluster_df
    # Optional comparison over several K values
    compare_fig = go.Figure()
    compare_rows = []
    k_list = sorted(set([int(k)] + [int(v.strip()) for v in str(compare_k).split(",") if v.strip().isdigit() and 2 <= int(v.strip()) <= 10]))
    for kval in k_list:
        if kval >= len(DATA):
            continue
        kmx = KMeans(n_clusters=kval, random_state=42, n_init=10)
        labx = kmx.fit_predict(X)
        try:
            sx = silhouette_score(X, labx, sample_size=min(1500, len(labx)), random_state=42)
        except Exception:
            sx = np.nan
        compare_rows.append({"K": kval, "Inertia": kmx.inertia_, "Silhouette": sx})
    compare_df = pd.DataFrame(compare_rows).sort_values("K")
    if not compare_df.empty:
        compare_fig.add_trace(go.Scatter(x=compare_df["K"], y=compare_df["Inertia"], mode="lines+markers", name="Inertia", yaxis="y"))
        compare_fig.add_trace(go.Scatter(x=compare_df["K"], y=compare_df["Silhouette"], mode="lines+markers", name="Silhouette", yaxis="y2"))
        compare_fig.update_layout(
            title="Compare K values",
            xaxis_title="Number of clusters (K)",
            yaxis=dict(title="Inertia"),
            yaxis2=dict(title="Silhouette score", overlaying="y", side="right"),
            template="plotly_white", height=400, title_x=0.02
        )
    else:
        compare_fig = px.line(title="No valid K values selected")
    return metrics, fig_sizes, fig_pca, compare_fig, assigned, profile

# ---- Classification comparative study ----
def classification_view(metric):
    comp = COMPARISON.copy()
    metric_map = {
        "Accuracy": "Accuracy (%)",
        "Precision": "Precision - Leave (%)",
        "Recall": "Recall - Leave (%)",
        "F1-score": "F1 - Leave (%)",
        "ROC-AUC": "ROC-AUC"
    }
    col = metric_map[metric]
    fig = px.bar(comp, x="Algorithm", y=col, color="Algorithm",
                 title=f"Algorithm comparison — {metric}")
    fig = nice_fig(fig, 390)
    # Confusion matrices for selected model output
    figs = []
    for name in ["Decision Tree", "Random Forest"]:
        cm = MODEL_EVAL[name]["confusion"]
        cmfig = px.imshow(
            cm, text_auto=True, x=["Predicted Stay", "Predicted Leave"],
            y=["Actual Stay", "Actual Leave"], color_continuous_scale="Blues",
            title=f"{name}: confusion matrix"
        )
        figs.append(nice_fig(cmfig, 350))
    return comp, fig, figs[0], figs[1]

# ---- Association rule mining ----
def mine_rules(min_support, min_confidence, max_len, columns, top_n, rank_by):
    if not columns or len(columns) < 2:
        raise gr.Error("Select at least two columns for rule mining.")
    if not 0 < float(min_support) <= 1:
        raise gr.Error("Support must be greater than 0 and at most 1.")
    if not 0 <= float(min_confidence) <= 1:
        raise gr.Error("Confidence must be between 0 and 1.")

    transaction_df = DATA[columns].copy()
    transactions = []
    for _, row in transaction_df.iterrows():
        transactions.append([
            f"{col}={('Unknown' if pd.isna(row[col]) else str(row[col]).strip())}"
            for col in columns
        ])
    encoder = TransactionEncoder()
    arr = encoder.fit(transactions).transform(transactions)
    basket = pd.DataFrame(arr, columns=encoder.columns_)
    itemsets = apriori(
        basket, min_support=float(min_support),
        use_colnames=True, max_len=int(max_len)
    )
    if itemsets.empty:
        return pd.DataFrame(), None, f"No frequent itemsets found. Lower support (currently {min_support:.1%}).", pd.DataFrame()
    try:
        rules = association_rules(
            itemsets, metric="confidence",
            min_threshold=float(min_confidence)
        )
    except Exception as e:
        return pd.DataFrame(), None, f"No rules generated: {e}", itemsets
    if rules.empty:
        return pd.DataFrame(), None, "No rules met the selected support and confidence thresholds.", itemsets
    rules["Antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["Consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    rules["Support (%)"] = (rules["support"] * 100).round(2)
    rules["Confidence (%)"] = (rules["confidence"] * 100).round(2)
    rules["Lift"] = rules["lift"].round(3)
    rules = rules.sort_values(rank_by.lower(), ascending=False) if rank_by.lower() in rules.columns else rules.sort_values("lift", ascending=False)
    output = rules[["Antecedents", "Consequents", "Support (%)", "Confidence (%)", "Lift"]].copy()
    top = output.head(int(top_n))
    fig = px.scatter(
        output, x="Support (%)", y="Confidence (%)",
        color="Lift", size="Lift", hover_data=["Antecedents", "Consequents"],
        title="Association rules: Support vs Confidence",
        color_continuous_scale="Viridis"
    )
    fig = nice_fig(fig, 450)
    status = f"Frequent itemsets: {len(itemsets):,} | Rules generated: {len(rules):,} | Displaying {len(top):,} rules ranked by {rank_by}."
    return top, fig, status, itemsets

# ---- Prediction ----
def predict_employee(model_name, threshold, age, travel, department, distance, education,
                     environment, gender, salary, involvement, job_level, job_role,
                     satisfaction, marital, companies, overtime, hike, experience,
                     worklife, years_company, years_role, promotion):
    values = {
        "Age": age, "Business_Travel": travel, "Department": department,
        "Distance_From_Home": distance, "Education": education,
        "Environment_Satisfaction": environment, "Gender": gender,
        "Salary": salary, "Job_Involvement": involvement, "Job_Level": str(job_level),
        "Job_Role": job_role, "Job_Satisfaction": satisfaction,
        "Marital_Status": marital,
        "Number_of_Companies_Worked_previously": companies,
        "Overtime": overtime, "Salary_Hike_in_percent": hike,
        "Total_working_years_experience": experience,
        "Work_life_balance": worklife,
        "No_of_years_worked_at_current_company": years_company,
        "No_of_years_in_current_role": years_role,
        "Years_since_last_promotion": promotion
    }
    row = {}
    for col in MODEL_FEATURES:
        val = values.get(col, DATA[col].mode(dropna=True).iloc[0] if not DATA[col].mode(dropna=True).empty else 0)
        row[col] = val
    input_df = pd.DataFrame([row], columns=MODEL_FEATURES)
    model = MODELS[model_name]
    prob = float(model.predict_proba(input_df)[0, list(model.classes_).index("Yes")])
    predicted = "Likely to Leave" if prob >= float(threshold) else "Likely to Stay"
    risk = "High" if prob >= 0.60 else ("Medium" if prob >= 0.30 else "Low")
    result_html = f"""
    <div class="prediction-card">
      <div class="kpi-label">Selected model: {model_name}</div>
      <div class="prediction-status">{predicted}</div>
      <div class="probability">{prob*100:.2f}%</div>
      <div>Estimated attrition probability</div>
      <div class="risk risk-{risk.lower()}">{risk} risk (project-defined bands)</div>
    </div>
    """
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100,
        number={"suffix": "%", "valueformat": ".2f"},
        title={"text": "Estimated probability of Attrition = Yes"},
        gauge={"axis":{"range":[0,100]},
               "steps":[{"range":[0,30],"color":"#dcfce7"},
                        {"range":[30,60],"color":"#fef3c7"},
                        {"range":[60,100],"color":"#fee2e2"}],
               "threshold":{"line":{"color":"#111827","width":3},"thickness":0.75,"value":float(threshold)*100}}
    ))
    return result_html, nice_fig(gauge, 350), pd.DataFrame([row])

# ---- Dashboard CSS ----
CSS = """
.gradio-container {max-width: 1500px !important; margin: auto !important;}
#hero {padding: 24px 28px; border-radius: 18px; background: linear-gradient(120deg,#111827,#1d4ed8); color: white; margin-bottom: 12px;}
#hero h1 {font-size: 30px; margin: 0 0 6px 0;}
#hero p {margin: 0; color: #dbeafe; font-size: 14px;}
.kpi-grid {display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:8px 0 16px;}
.kpi {padding:18px; border:1px solid #e5e7eb; border-radius:14px; background:#fff;}
.kpi-label {font-size:12px; color:#6b7280; margin-bottom:7px;}
.kpi-value {font-size:25px; font-weight:750; color:#111827;}
.prediction-card {padding:22px; border-radius:16px; background:#f8fafc; border:1px solid #e2e8f0; text-align:center;}
.prediction-status {font-size:26px; font-weight:750; margin:10px 0;}
.probability {font-size:42px; font-weight:800; color:#1d4ed8;}
.risk {display:inline-block; margin-top:12px; padding:7px 14px; border-radius:20px; font-weight:700;}
.risk-low {background:#dcfce7;color:#166534;}
.risk-medium {background:#fef3c7;color:#92400e;}
.risk-high {background:#fee2e2;color:#991b1b;}
"""

# ---- Gradio UI ----
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, title="Employee Attrition Analytics") as demo:
    gr.HTML("""
    <div id="hero">
      <h1>Employee Attrition Analytics</h1>
      <p>DWDM project · Clustering · Classification · Association Rules · Attrition Prediction</p>
    </div>
    """)
    gr.Markdown(
        f"**Dataset:** {len(DATA):,} employees · **Features:** {len(MODEL_FEATURES)} · "
        "**Models:** Decision Tree and Random Forest"
    )

    with gr.Tabs():
        with gr.Tab("Overview"):
            overview_cards = gr.HTML()
            with gr.Row():
                overview_pie = gr.Plot()
                overview_models = gr.Dataframe(label="Classification model comparison", interactive=False)
            gr.Button("Refresh overview").click(
                overview, outputs=[overview_cards, overview_pie, overview_models]
            )
            demo.load(overview, outputs=[overview_cards, overview_pie, overview_models])

        with gr.Tab("K-Means Clustering"):
            gr.Markdown("### Interactive employee segmentation\nAttrition is excluded from clustering inputs. PCA is used only to display the clusters in 2D.")
            with gr.Row():
                k_input = gr.Slider(2, 10, value=3, step=1, label="Number of clusters (K)")
                sample_input = gr.Slider(500, min(5000, len(DATA)), value=min(2000, len(DATA)), step=250, label="Points shown in 2D plot")
            cluster_features = gr.Dropdown(
                choices=CLUSTER_FEATURES, value=CLUSTER_FEATURES,
                multiselect=True, label="Features used for clustering"
            )
            compare_k_input = gr.Textbox(value="2,3,4,5,6,7,8", label="K values to compare (comma-separated)", info="The current K is always included.")
            run_cluster_btn = gr.Button("Run clustering", variant="primary")
            cluster_metrics = gr.HTML()
            with gr.Row():
                cluster_sizes_plot = gr.Plot()
                cluster_pca_plot = gr.Plot()
            cluster_comparison_plot = gr.Plot()
            with gr.Row():
                cluster_table = gr.Dataframe(label="Employee cluster assignments", interactive=False)
                cluster_profile = gr.Dataframe(label="Attrition distribution within clusters (%)", interactive=False)
            run_cluster_btn.click(
                run_clustering,
                inputs=[k_input, cluster_features, compare_k_input, sample_input],
                outputs=[cluster_metrics, cluster_sizes_plot, cluster_pca_plot,
                         cluster_comparison_plot, cluster_table, cluster_profile]
            )

        with gr.Tab("Classification & Comparison"):
            gr.Markdown("### Decision Tree vs Random Forest\nBoth models use the same stratified 80/20 train-test split and preprocessing. Recall is important when the goal is to identify employees who may leave.")
            metric_choice = gr.Dropdown(
                ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
                value="Recall", label="Metric to visualize"
            )
            with gr.Row():
                comparison_table = gr.Dataframe(value=COMPARISON, label="Comparative evaluation", interactive=False)
                comparison_plot = gr.Plot()
            with gr.Row():
                dt_cm_plot = gr.Plot()
                rf_cm_plot = gr.Plot()
            metric_choice.change(
                classification_view, inputs=[metric_choice],
                outputs=[comparison_table, comparison_plot, dt_cm_plot, rf_cm_plot]
            )
            demo.load(
                lambda: classification_view("Recall"),
                outputs=[comparison_table, comparison_plot, dt_cm_plot, rf_cm_plot]
            )

        with gr.Tab("Association Rule Mining"):
            gr.Markdown("### Apriori association rules\nChoose attributes and thresholds. Support and confidence are entered as percentages; the dashboard converts them to the Apriori fractions.")
            rule_cols = gr.Dropdown(
                choices=[c for c in DATA.columns if c != ID_COL],
                value=[c for c in ["Department", "Job_Role", "Overtime", "Job_Satisfaction", "Work_life_balance", "Marital_Status", "Attrition"] if c in DATA.columns],
                multiselect=True, label="Attributes included in transactions"
            )
            with gr.Row():
                support_input = gr.Slider(1, 30, value=2, step=1, label="Minimum support (%)")
                confidence_input = gr.Slider(1, 100, value=30, step=1, label="Minimum confidence (%)")
            with gr.Row():
                max_len_input = gr.Slider(2, 4, value=3, step=1, label="Maximum itemset length")
                topn_input = gr.Slider(10, 200, value=50, step=10, label="Maximum rules to display")
                rank_input = gr.Radio(["Lift", "Confidence", "Support"], value="Lift", label="Rank rules by")
            mine_btn = gr.Button("Mine association rules", variant="primary")
            rule_status = gr.Markdown("Set parameters and click **Mine association rules**.")
            with gr.Row():
                rule_table = gr.Dataframe(label="Association rules", interactive=False)
                rule_plot = gr.Plot()
            rule_download = gr.DownloadButton(label="Download displayed rules as CSV")
            def mine_and_export(sup, conf, max_len, cols, topn, rank):
                table, fig, status, itemsets = mine_rules(
                    sup/100, conf/100, max_len, cols, topn, rank
                )
                path = None
                if table is not None and not table.empty:
                    path = "/tmp/association_rules_dashboard.csv"
                    table.to_csv(path, index=False)
                return table, fig, status, path
            mine_btn.click(
                mine_and_export,
                inputs=[support_input, confidence_input, max_len_input, rule_cols, topn_input, rank_input],
                outputs=[rule_table, rule_plot, rule_status, rule_download]
            )

        with gr.Tab("Attrition Prediction"):
            gr.Markdown("### Predict attrition for an employee\nSelect a model and enter employee details. Probabilities are model estimates, not certainties. Risk bands follow the project's illustrative thresholds: Low <30%, Medium 30–<60%, High ≥60%.")
            with gr.Row():
                pred_model = gr.Radio(["Random Forest", "Decision Tree"], value="Random Forest", label="Prediction method")
                pred_threshold = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Decision threshold")
            with gr.Row():
                p_age = gr.Number(value=35, label="Age")
                p_travel = gr.Dropdown(choices=sorted(DATA["Business_Travel"].dropna().astype(str).unique()) if "Business_Travel" in DATA else [], label="Business Travel")
                p_dept = gr.Dropdown(choices=sorted(DATA["Department"].dropna().astype(str).unique()) if "Department" in DATA else [], label="Department")
                p_distance = gr.Number(value=10, label="Distance From Home")
            with gr.Row():
                p_education = gr.Dropdown(choices=sorted(DATA["Education"].dropna().astype(str).unique()) if "Education" in DATA else [], label="Education")
                p_env = gr.Number(value=3, label="Environment Satisfaction")
                p_gender = gr.Dropdown(choices=sorted(DATA["Gender"].dropna().astype(str).unique()) if "Gender" in DATA else [], label="Gender")
                p_salary = gr.Number(value=float(DATA["Salary"].median()) if "Salary" in DATA else 50000, label="Salary")
            with gr.Row():
                p_involvement = gr.Number(value=3, label="Job Involvement")
                p_job_level = gr.Dropdown(choices=sorted(DATA["Job_Level"].dropna().astype(str).unique()) if "Job_Level" in DATA else ["1"], value=str(DATA["Job_Level"].mode().iloc[0]) if "Job_Level" in DATA else "1", label="Job Level")
                p_role = gr.Dropdown(choices=sorted(DATA["Job_Role"].dropna().astype(str).unique()) if "Job_Role" in DATA else [], label="Job Role")
                p_satisfaction = gr.Number(value=3, label="Job Satisfaction")
            with gr.Row():
                p_marital = gr.Dropdown(choices=sorted(DATA["Marital_Status"].dropna().astype(str).unique()) if "Marital_Status" in DATA else [], label="Marital Status")
                p_companies = gr.Number(value=2, label="Previous Companies")
                p_overtime = gr.Dropdown(choices=sorted(DATA["Overtime"].dropna().astype(str).unique()) if "Overtime" in DATA else [], label="Overtime")
                p_hike = gr.Number(value=float(DATA["Salary_Hike_in_percent"].median()) if "Salary_Hike_in_percent" in DATA else 15, label="Salary Hike (%)")
            with gr.Row():
                p_experience = gr.Number(value=10, label="Total Working Years Experience")
                p_worklife = gr.Number(value=3, label="Work-Life Balance")
                p_years_company = gr.Number(value=3, label="Years at Current Company")
                p_years_role = gr.Number(value=2, label="Years in Current Role")
                p_promotion = gr.Number(value=1, label="Years Since Last Promotion")
            predict_btn = gr.Button("Predict attrition risk", variant="primary")
            prediction_result = gr.HTML()
            prediction_gauge = gr.Plot()
            prediction_input_table = gr.Dataframe(label="Input used for prediction", interactive=False)
            predict_btn.click(
                predict_employee,
                inputs=[
                    pred_model, pred_threshold, p_age, p_travel, p_dept, p_distance,
                    p_education, p_env, p_gender, p_salary, p_involvement, p_job_level,
                    p_role, p_satisfaction, p_marital, p_companies, p_overtime, p_hike,
                    p_experience, p_worklife, p_years_company, p_years_role, p_promotion
                ],
                outputs=[prediction_result, prediction_gauge, prediction_input_table]
            )

    gr.Markdown(
        "### Notes\n"
        "- Clustering excludes Attrition and Employee_ID from its input features.\n"
        "- Classification models are evaluated on a held-out test set; metrics may vary if the dataset changes.\n"
        "- Association rules describe co-occurrence, not causation.\n"
        "- Prediction probabilities are model estimates and should support—not replace—human review."
    )

print("Launching dashboard. Generating public Gradio URL...")
app_obj, local_url, share_url = demo.launch(share=True, server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
print(f"GRADIO_LOCAL_URL: {local_url}")
print(f"GRADIO_SHARE_URL: {share_url}")
url_path = os.path.join(BASE_DIR, "gradio_public_url.txt")
with open(url_path, "w", encoding="utf-8") as f:
    f.write(str(share_url or ""))

if share_url:
    index_html_path = os.path.join(BASE_DIR, "netlify-site", "index.html")
    if os.path.exists(index_html_path):
        with open(index_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        import re
        updated_content = re.sub(r'https://[a-zA-Z0-9-]+\.gradio\.live', share_url, content)
        with open(index_html_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print("Updated netlify-site/index.html with live Gradio URL.")
