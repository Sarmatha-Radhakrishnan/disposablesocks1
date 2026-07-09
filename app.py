"""
Disposable Active Socks — End-to-End Analytics & ML Dashboard
Group 3 | Streamlit App

Sections:
 1. Descriptive Analytics
 2. Diagnostic Analytics
 3. Segmentation (K-Means, Hierarchical/Dendrogram, RFM, Regression)
 4. Classification (KNN, Decision Tree, Random Forest, Gradient Boosting)
 5. Anomaly Detection
 6. Association Rule Mining (Market Basket / Bundle Analysis)
 7. Prescriptive Insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.preprocessing import label_binarize

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Disposable Active Socks — Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUNDLE_ITEM_COLS = {
    "Owns_Gym_Bag": "Gym_Bag",
    "Owns_Shaker_Bottle": "Shaker_Bottle",
    "Buys_Sweat_Towel": "Sweat_Towel",
    "Buys_Body_Wipes": "Body_Wipes",
    "Buys_Deodorant_Sticks": "Deodorant_Sticks",
    "Interested_Disposable_Underwear": "Disposable_Underwear",
    "Uses_Foot_Powder_Insert": "Foot_Powder_Insert",
    "Buys_Blister_Plasters": "Blister_Plasters",
    "Uses_Muscle_Recovery_Balm": "Muscle_Recovery_Balm",
}

DEMO_BEHAVIOR_FEATURES = [
    "Age", "Gym_Days_Per_Week", "Gym_Duration_Years", "Hygiene_Score",
    "Forgetting_Freq", "Monthly_Activewear_Spend", "Subscription_Interest",
    "Eco_Trend_Score",
]


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/disposable_socks_data.csv")
    return df


df = load_data()


# ----------------------------------------------------------------------------
# CACHED COMPUTE HELPERS
# Streamlit reruns the whole script on every widget interaction. Without
# caching, dragging one slider re-trains every model on the page. These
# functions cache on their actual inputs so a click only recomputes what
# actually changed.
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Running K-Means for k = 2..10...")
def compute_kmeans_elbow(feature_tuple):
    X_scaled = StandardScaler().fit_transform(df[list(feature_tuple)])
    inertias, sil_scores = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, km.labels_))
    return inertias, sil_scores


@st.cache_data(show_spinner="Fitting final K-Means model...")
def compute_kmeans_final(feature_tuple, k):
    X_scaled = StandardScaler().fit_transform(df[list(feature_tuple)])
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    return labels, pcs, pca.explained_variance_ratio_.sum()


@st.cache_data(show_spinner="Training classification models...")
def train_classifiers(feature_tuple, include_cat, test_size):
    X = df[list(feature_tuple)].copy()
    if include_cat:
        for c in ["Gender", "Income", "Occupation"]:
            X[c] = LabelEncoder().fit_transform(df[c])
    y = df["WTP_Disposable_Socks"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
    }

    results, roc_data, cm_data = [], {}, {}
    rf_importance = None
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        train_acc = accuracy_score(y_train, model.predict(X_train_s))
        test_acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        roc_data[name] = (fpr, tpr, roc_auc)
        cm_data[name] = confusion_matrix(y_test, y_pred)
        results.append({
            "Model": name, "Train Accuracy": round(train_acc, 3), "Test Accuracy": round(test_acc, 3),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
            "F1 Score": round(f1_score(y_test, y_pred, zero_division=0), 3),
            "ROC AUC": round(roc_auc, 3),
        })
        if name == "Random Forest":
            rf_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return pd.DataFrame(results), roc_data, cm_data, rf_importance


@st.cache_data(show_spinner="Detecting anomalies...")
def compute_anomalies(feature_tuple, contamination):
    X = df[list(feature_tuple)].copy()
    if "Max_WTP_Per_Pair" in feature_tuple:
        X["Max_WTP_Per_Pair"] = df["Max_WTP_Per_Pair"].map({"Under_15": 10, "15-25": 20, "25-35": 30, "35+": 40})
    X_scaled = StandardScaler().fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)
    pcs = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    return labels, scores, pcs


@st.cache_data(show_spinner="Mining association rules...")
def compute_association_rules(min_support, min_confidence):
    basket = pd.DataFrame()
    basket["Disposable_Socks"] = df["WTP_Disposable_Socks"] == 1
    for col, name in BUNDLE_ITEM_COLS.items():
        basket[name] = df[col] == "Yes"

    freq_items = apriori(basket, min_support=min_support, use_colnames=True, max_len=3)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values("lift", ascending=False)

    mask = rules["antecedents"].apply(lambda x: "Disposable_Socks" in x) | rules["consequents"].apply(lambda x: "Disposable_Socks" in x)
    socks_rules = rules[mask].copy()
    socks_rules["antecedents"] = socks_rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    socks_rules["consequents"] = socks_rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    socks_rules = socks_rules[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)

    supp_socks = basket["Disposable_Socks"].mean()
    recs = []
    for name in BUNDLE_ITEM_COLS.values():
        supp_p = basket[name].mean()
        supp_both = (basket["Disposable_Socks"] & basket[name]).mean()
        conf = supp_both / supp_socks
        lift = supp_both / (supp_socks * supp_p)
        recs.append({"Product": name, "Support": round(supp_p, 3), "Confidence (Socks→Product)": round(conf, 3), "Lift": round(lift, 3)})
    recs_df = pd.DataFrame(recs).sort_values("Lift", ascending=False).reset_index(drop=True)
    recs_df["Bundle_Tier"] = recs_df["Lift"].apply(lambda l: "Tier 1 (Strong)" if l >= 1.2 else ("Tier 2 (Moderate)" if l >= 1.0 else "Tier 3 (Weak)"))

    return socks_rules, recs_df

st.sidebar.title("🧦 Disposable Active Socks")
st.sidebar.caption("Group 3 — End-to-End Analytics Dashboard")
section = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 Descriptive Analytics",
        "🔍 Diagnostic Analytics",
        "🧩 Segmentation (Clustering & RFM)",
        "🤖 Classification Models",
        "🚨 Anomaly Detection",
        "🛒 Association Rule Mining",
        "💡 Prescriptive Insights",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {df.shape[0]} respondents · {df.shape[1]} variables")

# ============================================================================
# 0. OVERVIEW
# ============================================================================
if section == "🏠 Overview":
    st.title("Disposable Active Socks — Analytics & ML Dashboard")
    st.markdown(
        """
This dashboard implements the full analytics stack requested for the project brief:
**descriptive → diagnostic → predictive → prescriptive** analytics, plus the two
revisions suggested after the last review:

1. **Clustering now uses a focused, meaningful subset of variables** (demographic +
   behavioral) instead of all 31 raw columns, to avoid overlap/noise in K-Means.
2. **Association rule mining (market basket analysis)** has been added to identify
   which products are commonly purchased alongside disposable socks, with a
   recommendation to start with the **top 3–4 highest-lift items** rather than
   bundling all 9 candidate products at once.
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{df.shape[0]}")
    c2.metric("Willing to Pay (WTP)", f"{df['WTP_Disposable_Socks'].mean()*100:.1f}%")
    c3.metric("Avg. Monthly Activewear Spend", f"${df['Monthly_Activewear_Spend'].mean():.0f}")
    c4.metric("Subscription Interest (avg /5)", f"{df['Subscription_Interest'].mean():.1f}")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)

# ============================================================================
# 1. DESCRIPTIVE ANALYTICS
# ============================================================================
elif section == "📊 Descriptive Analytics":
    st.title("📊 Descriptive Analytics")
    st.caption("What does the data look like? Summary stats, distributions, cross-tabs.")

    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Distributions", "Cross-Tabulation"])

    with tab1:
        st.subheader("Numeric Summary")
        st.dataframe(df.describe().T, use_container_width=True)
        st.subheader("Categorical Summary")
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        cat_col = st.selectbox("Choose a categorical variable", cat_cols, index=cat_cols.index("Primary_Pain_Point") if "Primary_Pain_Point" in cat_cols else 0)
        vc = df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, "Count"]
        col_a, col_b = st.columns([1, 1])
        col_a.dataframe(vc, use_container_width=True)
        fig = px.pie(vc, names=cat_col, values="Count", title=f"Distribution of {cat_col}")
        col_b.plotly_chart(fig, use_container_width=True)

    with tab2:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        num_col = st.selectbox("Choose a numeric variable", num_cols, index=num_cols.index("Monthly_Activewear_Spend") if "Monthly_Activewear_Spend" in num_cols else 0)
        fig = px.histogram(df, x=num_col, nbins=30, marginal="box", title=f"Distribution of {num_col}")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Cross-Tabulation")
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        c1, c2 = st.columns(2)
        var1 = c1.selectbox("Row variable", cat_cols, index=cat_cols.index("Occupation") if "Occupation" in cat_cols else 0)
        var2 = c2.selectbox("Column variable", cat_cols, index=cat_cols.index("Primary_Pain_Point") if "Primary_Pain_Point" in cat_cols else 1)
        ctab = pd.crosstab(df[var1], df[var2], normalize="index").round(3) * 100
        st.dataframe(ctab, use_container_width=True)
        fig = px.imshow(ctab, text_auto=True, aspect="auto", labels=dict(color="% of row"), title=f"{var1} vs {var2} (% within row)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 2. DIAGNOSTIC ANALYTICS
# ============================================================================
elif section == "🔍 Diagnostic Analytics":
    st.title("🔍 Diagnostic Analytics")
    st.caption("Why is it happening? Relationships, correlations, and group-level bias checks.")

    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Group Comparison", "Bias / Fairness Check"])

    with tab1:
        num_df = df.select_dtypes(include=np.number).drop(columns=["Response_ID"], errors="ignore")
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlation Matrix — Numeric Variables")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        c1, c2 = st.columns(2)
        group_var = c1.selectbox("Group by", cat_cols, index=cat_cols.index("Income") if "Income" in cat_cols else 0)
        metric_var = c2.selectbox("Compare metric", num_cols, index=num_cols.index("Monthly_Activewear_Spend") if "Monthly_Activewear_Spend" in num_cols else 0)
        fig = px.box(df, x=group_var, y=metric_var, color=group_var, title=f"{metric_var} by {group_var}")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.groupby(group_var)[metric_var].agg(["mean", "median", "std", "count"]).round(2), use_container_width=True)

    with tab3:
        st.markdown("Checking whether WTP / spend behaves unevenly across sensitive groups (Gender, Income, Occupation).")
        sensitive_var = st.selectbox("Sensitive attribute", ["Gender", "Income", "Occupation"])
        bias_tab = df.groupby(sensitive_var).agg(
            WTP_Rate=("WTP_Disposable_Socks", "mean"),
            Avg_Spend=("Monthly_Activewear_Spend", "mean"),
            Avg_Subscription_Interest=("Subscription_Interest", "mean"),
        ).round(3)
        st.dataframe(bias_tab, use_container_width=True)
        fig = px.bar(bias_tab.reset_index(), x=sensitive_var, y="WTP_Rate", title=f"WTP Rate across {sensitive_var} groups")
        st.plotly_chart(fig, use_container_width=True)
        spread = bias_tab["WTP_Rate"].max() - bias_tab["WTP_Rate"].min()
        if spread > 0.15:
            st.warning(f"⚠️ WTP rate varies by {spread*100:.1f} percentage points across {sensitive_var} groups — worth investigating for uneven targeting/bias before building marketing strategy.")
        else:
            st.success(f"✅ WTP rate is relatively even across {sensitive_var} groups (spread = {spread*100:.1f} pts).")

# ============================================================================
# 3. SEGMENTATION: K-MEANS, DENDROGRAM, RFM, REGRESSION
# ============================================================================
elif section == "🧩 Segmentation (Clustering & RFM)":
    st.title("🧩 Segmentation")
    st.caption("Customer segmentation using a focused variable set — addressing the feedback to avoid dumping all 31 raw columns into K-Means.")

    tab1, tab2, tab3, tab4 = st.tabs(["K-Means Clustering", "Hierarchical / Dendrogram", "RFM Analysis", "Regression (Lasso/Ridge/Linear)"])

    # ---- K-MEANS ----
    with tab1:
        st.subheader("Feature Selection")
        st.info("Per instructor feedback: instead of all 31 variables, we use a curated set of demographic + behavioral features that are most meaningful for segmentation.")
        selected_features = st.multiselect(
            "Clustering features (demographic + behavioral)",
            DEMO_BEHAVIOR_FEATURES,
            default=DEMO_BEHAVIOR_FEATURES[:6],
        )
        if len(selected_features) < 2:
            st.warning("Select at least 2 features.")
        else:
            feat_tuple = tuple(selected_features)

            st.subheader("Elbow Method & Silhouette Score")
            inertias, sil_scores = compute_kmeans_elbow(feat_tuple)
            k_range = range(2, 11)

            c1, c2 = st.columns(2)
            fig_elbow = px.line(x=list(k_range), y=inertias, markers=True, labels={"x": "k (clusters)", "y": "Inertia (WCSS)"}, title="Elbow Method")
            c1.plotly_chart(fig_elbow, use_container_width=True)
            fig_sil = px.line(x=list(k_range), y=sil_scores, markers=True, labels={"x": "k (clusters)", "y": "Silhouette Score"}, title="Silhouette Score by k")
            c2.plotly_chart(fig_sil, use_container_width=True)

            best_k = list(k_range)[int(np.argmax(sil_scores))]
            st.success(f"Suggested optimal k (highest silhouette score) = **{best_k}**")
            chosen_k = st.slider("Choose number of clusters (k)", 2, 10, best_k)

            cluster_labels, pcs, explained_var = compute_kmeans_final(feat_tuple, chosen_k)
            df["Cluster"] = cluster_labels

            st.subheader("Cluster Profiles (mean values)")
            profile = df.groupby("Cluster")[selected_features].mean().round(2)
            profile["Segment_Size"] = df["Cluster"].value_counts().sort_index()
            st.dataframe(profile, use_container_width=True)

            st.subheader("3D Cluster Visualization (PCA-reduced)")
            plot_df = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"])
            plot_df["Cluster"] = df["Cluster"].astype(str)
            fig3d = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color="Cluster",
                                  title=f"3D PCA View of {chosen_k} Clusters (explained variance = {explained_var*100:.1f}%)")
            st.plotly_chart(fig3d, use_container_width=True)

            st.markdown(
                f"""
**Interpretation:** With {chosen_k} clusters on the selected demographic/behavioral
features, segments typically separate along spend level, gym frequency, and hygiene
consciousness — e.g. a high-frequency / high-hygiene "premium athlete" group, a
moderate "corporate casual" group, and a lower-spend "budget-conscious" group.
Use the cluster profile table above to name each segment based on your own data.
                """
            )

    # ---- DENDROGRAM ----
    with tab2:
        st.subheader("Hierarchical Clustering — Dendrogram")
        st.caption("Visualize how observations merge under different linkage methods.")
        linkage_method = st.selectbox("Linkage method", ["ward", "single", "complete", "average"])
        sample_n = st.slider("Sample size for dendrogram (for readability)", 30, 150, 60, step=10)

        hc_features = st.multiselect(
            "Features for hierarchical clustering",
            DEMO_BEHAVIOR_FEATURES,
            default=DEMO_BEHAVIOR_FEATURES[:5],
            key="hc_feats",
        )
        if len(hc_features) >= 2:
            sample_df = df[hc_features].sample(sample_n, random_state=42)
            X_hc = StandardScaler().fit_transform(sample_df)
            Z = linkage(X_hc, method=linkage_method)

            fig = ff.create_dendrogram(X_hc, linkagefun=lambda x: linkage(x, method=linkage_method))
            fig.update_layout(title=f"Dendrogram — {linkage_method.title()} Linkage (n={sample_n})", height=550)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**Linkage guide:**
- **Single linkage** — merges based on nearest points; can create long "chained" clusters, sensitive to noise.
- **Complete linkage** — merges based on farthest points; tends to create compact, evenly-sized clusters.
- **Average linkage** — a balance between single and complete; less sensitive to outliers.
- **Ward's linkage** — minimizes within-cluster variance; usually gives the most interpretable, evenly-sized segments and is a good default for business segmentation.
                """
            )

            n_clust_hc = st.slider("Cut dendrogram into how many clusters?", 2, 8, 3)
            agg = AgglomerativeClustering(n_clusters=n_clust_hc, linkage=linkage_method)
            hc_labels = agg.fit_predict(X_hc)
            st.write(f"Cluster sizes: {pd.Series(hc_labels).value_counts().sort_index().to_dict()}")
        else:
            st.warning("Select at least 2 features.")

    # ---- RFM ----
    with tab3:
        st.subheader("RFM Analysis (Recency, Frequency, Monetary)")
        st.caption("RFM variables built from purchase-recency, purchase-frequency (last 6 months), and spend behavior, to segment customers by value.")

        rfm = df[["Response_ID", "Recency_Days", "Frequency_Purchases_6M", "Monetary_Spend_6M"]].copy()
        rfm["R_Score"] = pd.qcut(rfm["Recency_Days"], 4, labels=[4, 3, 2, 1]).astype(int)
        rfm["F_Score"] = pd.qcut(rfm["Frequency_Purchases_6M"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["M_Score"] = pd.qcut(rfm["Monetary_Spend_6M"], 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

        def rfm_segment(score):
            if score >= 10:
                return "Champions"
            elif score >= 8:
                return "Loyal Customers"
            elif score >= 6:
                return "Potential Loyalists"
            elif score >= 4:
                return "At Risk"
            else:
                return "Lost / Low Value"

        rfm["Segment"] = rfm["RFM_Score"].apply(rfm_segment)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(rfm.head(15), use_container_width=True)
        with c2:
            seg_counts = rfm["Segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig = px.bar(seg_counts, x="Segment", y="Count", color="Segment", title="RFM Segment Distribution")
            st.plotly_chart(fig, use_container_width=True)

        fig3d = px.scatter_3d(rfm, x="Recency_Days", y="Frequency_Purchases_6M", z="Monetary_Spend_6M",
                               color="Segment", title="RFM 3D View")
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown(
            """
**How to read this:** Champions (recent, frequent, high spend) are prime candidates
for the premium bundle and subscription push. At Risk / Lost segments are better
targets for reactivation offers or price-sensitive single-pack options.
            """
        )

    # ---- REGRESSION ----
    with tab4:
        st.subheader("Regression: Linear vs Lasso vs Ridge")
        st.caption("Predicting a continuous target (Monthly Activewear Spend) to compare regularization approaches.")

        reg_features = st.multiselect(
            "Predictor variables",
            [c for c in DEMO_BEHAVIOR_FEATURES if c != "Monthly_Activewear_Spend"] + ["Imp_Price", "Imp_Brand", "Vendor_Trust_Score", "QC_Score"],
            default=["Age", "Gym_Days_Per_Week", "Hygiene_Score", "Subscription_Interest", "Imp_Price", "Vendor_Trust_Score"],
        )
        target = "Monthly_Activewear_Spend"

        if len(reg_features) >= 1:
            X = df[reg_features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            alpha = st.slider("Regularization strength (alpha) for Lasso/Ridge", 0.01, 10.0, 1.0)

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso Regression": Lasso(alpha=alpha),
                "Ridge Regression": Ridge(alpha=alpha),
            }

            results = []
            coef_table = pd.DataFrame(index=reg_features)
            for name, model in models.items():
                model.fit(X_train_s, y_train)
                train_r2 = model.score(X_train_s, y_train)
                test_r2 = model.score(X_test_s, y_test)
                results.append({"Model": name, "Train R²": round(train_r2, 3), "Test R²": round(test_r2, 3)})
                coef_table[name] = model.coef_

            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.subheader("Coefficient Comparison")
            st.dataframe(coef_table.round(3), use_container_width=True)
            fig = px.bar(coef_table.reset_index().melt(id_vars="index"), x="index", y="value", color="variable",
                         barmode="group", title="Feature Coefficients Across Models", labels={"index": "Feature", "value": "Coefficient"})
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(
                """
**Interpretation:** Lasso can shrink less-important feature coefficients to exactly
zero (built-in feature selection), while Ridge shrinks all coefficients smoothly
without eliminating them — useful when features are correlated. Compare Test R²
across the three to see which generalizes best on this dataset.
                """
            )
        else:
            st.warning("Select at least 1 predictor variable.")

# ============================================================================
# 4. CLASSIFICATION MODELS
# ============================================================================
elif section == "🤖 Classification Models":
    st.title("🤖 Predictive Modeling — Classification")
    st.caption("Predicting Willingness-to-Pay (WTP_Disposable_Socks) using KNN, Decision Tree, Random Forest, and Gradient Boosting.")

    target = "WTP_Disposable_Socks"
    # NOTE: Gym_Duration_Years is intentionally excluded — it is a near-perfect
    # deterministic proxy for the target in this synthetic dataset (data leakage)
    # and would make every model look artificially perfect (~100% accuracy).
    candidate_features = [
        "Age", "Gym_Days_Per_Week", "Hygiene_Score", "Forgetting_Freq",
        "Monthly_Activewear_Spend", "Subscription_Interest", "Eco_Trend_Score", "Vendor_Trust_Score",
        "QC_Score", "Imp_Price", "Imp_Brand", "Imp_Material", "Imp_Antibacterial", "Imp_Antimicrobial",
        "Imp_Lifestyle_Fit",
    ]

    st.subheader("Feature Engineering")
    selected = st.multiselect("Select features for classification", candidate_features, default=candidate_features[:8])

    # NOTE: Workout_Type is intentionally excluded — in this synthetic dataset
    # it is a deterministic proxy for the target (every "Weights" respondent
    # = No, every other type = Yes), which would make accuracy artificially
    # hit 100% regardless of the model used.
    include_cat = st.checkbox("Include encoded categorical features (Gender, Income, Occupation)", value=True)

    if len(selected) < 2:
        st.warning("Select at least 2 features.")
    else:
        test_size = st.slider("Test set size (%)", 10, 40, 25) / 100

        res_df, roc_data, cm_data, rf_importance = train_classifiers(tuple(selected), include_cat, test_size)
        res_df = res_df.sort_values("ROC AUC", ascending=False)

        st.subheader("Model Comparison")
        st.dataframe(res_df, use_container_width=True)

        best_model_name = res_df.iloc[0]["Model"]
        st.success(f"🏆 Best performing model on this run: **{best_model_name}** (ROC AUC = {res_df.iloc[0]['ROC AUC']})")

        if res_df["Test Accuracy"].min() > 0.9:
            st.caption(
                "ℹ️ Note: accuracy is very high across all models because the survey's attitudinal "
                "variables (price sensitivity, hygiene score, vendor trust, brand importance) were "
                "designed to move together with purchase intent — a known property of this synthetic "
                "dataset. Treat these numbers as an upper bound and validate with real transaction "
                "data before using them to size a business case."
            )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ROC Curves")
            fig = go.Figure()
            for name, (fpr, tpr, roc_auc) in roc_data.items():
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={roc_auc:.2f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Random"))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=450)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Confusion Matrix")
            cm_model = st.selectbox("Select model", list(cm_data.keys()))
            cm = cm_data[cm_model]
            fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                             x=["No", "Yes"], y=["No", "Yes"], title=f"Confusion Matrix — {cm_model}")
            st.plotly_chart(fig, use_container_width=True)

        if rf_importance is not None:
            st.subheader("Feature Importance (Random Forest)")
            fig = px.bar(rf_importance, orientation="h", title="Random Forest Feature Importances")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 5. ANOMALY DETECTION
# ============================================================================
elif section == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.caption("Identifying rare / risky respondents whose behavior sharply deviates from the norm — e.g. inconsistent WTP/spend patterns.")

    anomaly_features = st.multiselect(
        "Features to check for anomalies",
        DEMO_BEHAVIOR_FEATURES + ["Vendor_Trust_Score", "QC_Score", "Max_WTP_Per_Pair"],
        default=["Monthly_Activewear_Spend", "Gym_Days_Per_Week", "Hygiene_Score", "Subscription_Interest"],
    )
    contamination = st.slider("Expected anomaly proportion (contamination)", 0.01, 0.20, 0.05)

    if len(anomaly_features) >= 2:
        labels, scores, pcs = compute_anomalies(tuple(anomaly_features), contamination)
        df["Anomaly"] = labels
        df["Anomaly_Score"] = scores
        df["Anomaly_Label"] = df["Anomaly"].map({1: "Normal", -1: "Anomaly"})

        c1, c2 = st.columns([1, 2])
        c1.metric("Anomalies Detected", int((df["Anomaly"] == -1).sum()))
        c1.metric("% of Respondents", f"{(df['Anomaly']==-1).mean()*100:.1f}%")

        plot_df = pd.DataFrame(pcs, columns=["PC1", "PC2"])
        plot_df["Anomaly_Label"] = df["Anomaly_Label"].values
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Anomaly_Label",
                          color_discrete_map={"Normal": "#4C78A8", "Anomaly": "#E45756"},
                          title="Anomaly Detection (PCA-reduced view)")
        c2.plotly_chart(fig, use_container_width=True)

        st.subheader("Flagged Anomalous Respondents")
        st.dataframe(
            df[df["Anomaly"] == -1][["Response_ID"] + anomaly_features + ["Anomaly_Score"]].sort_values("Anomaly_Score"),
            use_container_width=True,
        )
        st.markdown(
            """
**Why this matters (constraints/definition of "risky"):** These respondents combine
unusual value patterns — e.g. very high spend with very low gym frequency, or high
subscription interest with very low hygiene concern — that don't fit the typical
customer archetype. They shouldn't drive segmentation or pricing decisions on their
own, but are worth a manual review (possible survey noise, or a genuinely unusual
customer type worth a follow-up interview).
            """
        )
    else:
        st.warning("Select at least 2 features.")

# ============================================================================
# 6. ASSOCIATION RULE MINING / MARKET BASKET
# ============================================================================
elif section == "🛒 Association Rule Mining":
    st.title("🛒 Association Rule Mining — Bundle Analysis")
    st.caption("Which products are commonly bought alongside disposable socks? Directly addresses the instructor's feedback on bundling.")

    min_support = st.slider("Minimum support", 0.02, 0.5, 0.05, step=0.01)
    min_confidence = st.slider("Minimum confidence", 0.1, 0.9, 0.5, step=0.05)

    socks_rules, recs_df = compute_association_rules(min_support, min_confidence)

    st.subheader(f"Association Rules Involving Disposable Socks ({len(socks_rules)} found)")
    st.dataframe(socks_rules.head(30), use_container_width=True)

    st.subheader("Single-Item Bundle Recommendation (Socks → Product)")
    c1, c2 = st.columns([1, 1])
    c1.dataframe(recs_df, use_container_width=True)
    fig = px.bar(recs_df, x="Product", y="Lift", color="Bundle_Tier", title="Lift by Candidate Bundle Product")
    fig.add_hline(y=1.0, line_dash="dash", annotation_text="Lift = 1 (no association)")
    c2.plotly_chart(fig, use_container_width=True)

    top4 = recs_df.head(4)["Product"].tolist()
    st.success(
        f"**Recommended starter bundle (per instructor's 'start with 3–4 items' guidance):** "
        f"Disposable Socks + {' + '.join(top4)}. Validate demand on this bundle before expanding to the remaining candidates."
    )

# ============================================================================
# 7. PRESCRIPTIVE INSIGHTS
# ============================================================================
elif section == "💡 Prescriptive Insights":
    st.title("💡 Prescriptive Insights")
    st.caption("Actionable recommendations synthesized from the analysis above.")

    st.markdown(
        f"""
### 1. Product Bundle Strategy
Start with the **top 3–4 items by lift** from the Association Rule Mining tab
(commonly: Body Wipes, Deodorant Sticks, Sweat Towel) alongside disposable socks.
Avoid launching all 9 candidate add-ons at once — validate demand on the core bundle
first, then expand.

### 2. Segment-Based Marketing
Use the **K-Means / RFM segments** to prioritize:
- **Champions / high-spend, high-frequency segment** → premium subscription push,
  highest average willingness-to-pay (~${df['Max_WTP_Per_Pair'].mode()[0] if 'Max_WTP_Per_Pair' in df.columns else 'N/A'} range).
- **At Risk / Lost segment** → reactivation discounts or single-pack trial offers
  rather than a subscription ask.

### 3. Pricing Guidance
{df['WTP_Disposable_Socks'].mean()*100:.0f}% of respondents are willing to pay, with the majority
comfortable in the $15–25 per 100-pack range (from the survey). Use this as the
anchor price point for the initial launch SKU.

### 4. Risk Monitoring
The Anomaly Detection tab flags a small set of respondents (~5%) with inconsistent
value patterns — monitor this segment rather than optimizing pricing/marketing
around them.

### 5. Model-Driven Targeting
The best-performing classifier (see Classification tab) can be used to score new
leads/survey respondents on **predicted willingness-to-pay**, letting the team
prioritize outreach instead of marketing to the full addressable list uniformly.
        """
    )

    st.info("Tip: Re-run the Classification and Segmentation tabs after each new data collection round to keep these recommendations current.")
