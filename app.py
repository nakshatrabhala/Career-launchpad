
import io
import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(
    page_title="Career App Founder Analytics",
    page_icon="📊",
    layout="wide",
)

DEFAULT_FILE = "uae_student_career_app_synthetic_data.xlsx"

TARGET_CLASS = "interest_in_app"
TARGET_REG = "monthly_expenditure_aed"

BINARY_GROUPS = {
    "Challenges": [
        "challenge_lack_of_skills",
        "challenge_no_guidance",
        "challenge_poor_resume",
        "challenge_no_connections",
        "challenge_lack_of_opportunities",
        "challenge_interview_fear",
    ],
    "Preferred Features": [
        "feature_resume_builder",
        "feature_internship_listings",
        "feature_mock_interviews",
        "feature_skill_courses",
        "feature_networking",
        "feature_job_alerts",
    ],
    "Skills": [
        "skill_excel",
        "skill_power_bi",
        "skill_python",
        "skill_communication",
        "skill_finance_accounting",
        "skill_marketing",
    ],
    "Platforms": [
        "platform_linkedin",
        "platform_internshala",
        "platform_naukri",
        "platform_indeed",
        "platform_college_placement_cell",
    ],
    "Missing in Current Platforms": [
        "missing_personalized_guidance",
        "missing_verified_opportunities",
        "missing_affordable_courses",
        "missing_mentor_access",
        "missing_interview_preparation",
        "missing_application_tracking",
    ],
}

CATEGORICAL_SINGLE = [
    "gender",
    "education_level",
    "field_of_study",
    "current_status",
    "current_year",
    "search_frequency",
    "pricing_preference",
    "learning_preference",
    "willingness_to_spend_monthly",
]

LIKERT_NUMERIC = [
    "career_goal_clarity",
    "importance_resume_builder",
    "importance_interview_practice",
    "importance_networking",
    "importance_job_tracking",
    "platform_satisfaction",
    "mentor_connection_likelihood",
]

NUMERIC_COLUMNS = [
    "age",
    "hours_skill_development_per_week",
    "monthly_expenditure_aed",
] + LIKERT_NUMERIC

MAPPING_LABELS = {
    "search_frequency": ["Rarely", "Monthly", "Weekly", "Daily"],
    "pricing_preference": ["Free features only", "Freemium", "Fully paid app"],
    "willingness_to_spend_monthly": ["AED 0", "AED 1-25", "AED 26-60", "AED 61+"],
    "current_year": ["1st Year", "2nd Year", "3rd Year", "Final Year", "Passed out"],
    "education_level": ["Undergraduate", "Postgraduate", "Diploma/Certification", "Other"],
    "current_status": ["Student", "Fresher", "Working professional"],
    "interest_in_app": ["No", "Maybe", "Yes"],
}

SEGMENT_RULES = {
    0: "Segment A",
    1: "Segment B",
    2: "Segment C",
    3: "Segment D",
}

@st.cache_data(show_spinner=False)
def load_default_dataset():
    if not os.path.exists(DEFAULT_FILE):
        st.error(f"Default dataset file '{DEFAULT_FILE}' not found in app directory.")
        st.stop()
    return read_dataset(DEFAULT_FILE)

def read_dataset(file_obj_or_path):
    if isinstance(file_obj_or_path, str):
        name = file_obj_or_path.lower()
    else:
        name = file_obj_or_path.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(file_obj_or_path)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xl = pd.ExcelFile(file_obj_or_path)
        if "Survey_Data" in xl.sheet_names:
            return pd.read_excel(file_obj_or_path, sheet_name="Survey_Data")
        return pd.read_excel(file_obj_or_path)
    raise ValueError("Unsupported file type. Please upload CSV or Excel.")

def coerce_binary_columns(df):
    out = df.copy()
    binary_cols = [c for cols in BINARY_GROUPS.values() for c in cols if c in out.columns]
    for col in binary_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        out[col] = (out[col] > 0).astype(int)
    return out

def clean_base_dataframe(df):
    out = df.copy()

    if "respondent_id" not in out.columns:
        out.insert(0, "respondent_id", range(1, len(out) + 1))

    out = coerce_binary_columns(out)

    for col in LIKERT_NUMERIC + ["age", "hours_skill_development_per_week", "monthly_expenditure_aed"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in CATEGORICAL_SINGLE + [TARGET_CLASS]:
        if col in out.columns:
            out[col] = out[col].astype("object").where(out[col].notna(), np.nan)

    return out

def summarize_binary_features(df):
    rows = []
    for group_name, cols in BINARY_GROUPS.items():
        for col in cols:
            if col in df.columns:
                rows.append(
                    {
                        "group": group_name,
                        "feature": col,
                        "selected_count": int(df[col].fillna(0).sum()),
                        "selected_pct": round(float(df[col].fillna(0).mean() * 100), 2),
                    }
                )
    return pd.DataFrame(rows)

def preprocess_for_modeling(df, include_targets=True):
    df = clean_base_dataframe(df)

    use_cols = [c for c in df.columns if c != "respondent_id"]
    work = df[use_cols].copy()

    if not include_targets:
        for target in [TARGET_CLASS, TARGET_REG]:
            if target in work.columns:
                work = work.drop(columns=[target])

    numeric_cols = [c for c in work.columns if work[c].dtype != "object"]
    categorical_cols = [c for c in work.columns if work[c].dtype == "object"]

    X_num = work[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=work.index)
    X_cat = work[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=work.index)

    if not X_num.empty:
        num_imputer = SimpleImputer(strategy="median")
        X_num = pd.DataFrame(num_imputer.fit_transform(X_num), columns=X_num.columns, index=X_num.index)

    if not X_cat.empty:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_cat = pd.DataFrame(cat_imputer.fit_transform(X_cat), columns=X_cat.columns, index=X_cat.index)
        X_cat = pd.get_dummies(X_cat, drop_first=False)

    X = pd.concat([X_num, X_cat], axis=1)
    return X

def align_new_data_to_training(new_df, training_columns):
    X_new = preprocess_for_modeling(new_df, include_targets=False)
    for col in training_columns:
        if col not in X_new.columns:
            X_new[col] = 0
    extra_cols = [c for c in X_new.columns if c not in training_columns]
    if extra_cols:
        X_new = X_new.drop(columns=extra_cols)
    X_new = X_new[training_columns]
    return X_new

@st.cache_resource(show_spinner=False)
def train_models(df):
    clean_df = clean_base_dataframe(df)

    model_df = clean_df.dropna(subset=[TARGET_CLASS]).copy()
    X_class = preprocess_for_modeling(model_df, include_targets=False)
    y_class = model_df[TARGET_CLASS].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X_class, y_class, test_size=0.25, random_state=42, stratify=y_class
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    labels = sorted(y_class.unique().tolist())
    y_test_bin = label_binarize(y_test, classes=labels)

    roc_records = []
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_records.append(
            {
                "label": label,
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc(fpr, tpr),
            }
        )

    perm = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    fi = pd.DataFrame(
        {"feature": X_test.columns, "importance": perm.importances_mean}
    ).sort_values("importance", ascending=False)

    class_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
    }

    reg_df = clean_df.dropna(subset=[TARGET_REG]).copy()
    X_reg = preprocess_for_modeling(reg_df, include_targets=False)
    y_reg = reg_df[TARGET_REG].astype(float)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.25, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(Xr_train, yr_train)
    yr_pred = reg.predict(Xr_test)

    reg_metrics = {
        "MAE": mean_absolute_error(yr_test, yr_pred),
        "RMSE": mean_squared_error(yr_test, yr_pred, squared=False),
        "R2": r2_score(yr_test, yr_pred),
        "actual_vs_pred": pd.DataFrame({"actual": yr_test, "predicted": yr_pred}),
    }

    cluster_df = clean_df.copy()
    X_cluster = preprocess_for_modeling(cluster_df, include_targets=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    sil_scores = []
    k_values = list(range(2, 7))
    for k in k_values:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = km_tmp.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels_tmp))

    optimal_k = k_values[int(np.argmax(sil_scores))]

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=optimal_k, random_state=42)
    latent_labels = gmm.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    pca_points = pca.fit_transform(X_scaled)
    cluster_plot = pd.DataFrame(
        {
            "pc1": pca_points[:, 0],
            "pc2": pca_points[:, 1],
            "cluster": cluster_labels.astype(str),
            "latent_class": latent_labels.astype(str),
        }
    )

    cluster_profile = cluster_df.copy()
    cluster_profile["cluster"] = cluster_labels
    profile_numeric_cols = [c for c in ["age", "career_goal_clarity", "hours_skill_development_per_week",
                                        "monthly_expenditure_aed", "platform_satisfaction",
                                        "mentor_connection_likelihood"] if c in cluster_profile.columns]
    cluster_summary = cluster_profile.groupby("cluster")[profile_numeric_cols].mean().round(2)

    interest_by_cluster = (
        cluster_profile.groupby(["cluster", TARGET_CLASS]).size().reset_index(name="count")
    )

    assoc_source_cols = [c for cols in BINARY_GROUPS.values() for c in cols if c in clean_df.columns]
    assoc_df = clean_df[assoc_source_cols].copy()
    assoc_df = assoc_df.fillna(0).astype(int)
    frequent_itemsets = apriori(assoc_df, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.35)
    if not rules.empty:
        rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
        rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
        rules = rules.sort_values(["lift", "confidence"], ascending=False)

    mi_scores = mutual_info_classif(
        X_class.fillna(0), y_class, discrete_features="auto", random_state=42
    )
    mi_df = pd.DataFrame({"feature": X_class.columns, "mutual_information": mi_scores}).sort_values(
        "mutual_information", ascending=False
    )

    cluster_names = generate_cluster_names(cluster_profile, optimal_k)

    return {
        "clean_df": clean_df,
        "class_model": clf,
        "class_training_columns": X_class.columns.tolist(),
        "class_metrics": class_metrics,
        "roc_records": roc_records,
        "feature_importance": fi,
        "mutual_info": mi_df,
        "reg_model": reg,
        "reg_training_columns": X_reg.columns.tolist(),
        "reg_metrics": reg_metrics,
        "cluster_scaler": scaler,
        "cluster_training_columns": X_cluster.columns.tolist(),
        "kmeans": kmeans,
        "gmm": gmm,
        "cluster_plot": cluster_plot,
        "cluster_summary": cluster_summary,
        "interest_by_cluster": interest_by_cluster,
        "optimal_k": optimal_k,
        "silhouette_scores": pd.DataFrame({"k": k_values, "silhouette_score": sil_scores}),
        "rules": rules,
        "cluster_names": cluster_names,
    }

def generate_cluster_names(cluster_profile, optimal_k):
    names = {}
    grouped = cluster_profile.groupby("cluster")
    for cluster_id in range(optimal_k):
        grp = grouped.get_group(cluster_id)
        avg_search = grp["search_frequency"].mode(dropna=True)
        top_feature = (
            grp[[c for c in BINARY_GROUPS["Preferred Features"] if c in grp.columns]].mean().sort_values(ascending=False)
        )
        top_challenge = (
            grp[[c for c in BINARY_GROUPS["Challenges"] if c in grp.columns]].mean().sort_values(ascending=False)
        )
        feature_name = top_feature.index[0].replace("feature_", "").replace("_", " ").title() if len(top_feature) else "General"
        challenge_name = top_challenge.index[0].replace("challenge_", "").replace("_", " ").title() if len(top_challenge) else "General"
        if grp["monthly_expenditure_aed"].mean() >= cluster_profile["monthly_expenditure_aed"].median():
            budget_tag = "Higher-Budget"
        else:
            budget_tag = "Budget-Sensitive"
        search_tag = avg_search.iloc[0] if not avg_search.empty else "Mixed Search"
        names[cluster_id] = f"{budget_tag} {search_tag} users needing {feature_name} / {challenge_name}"
    return names

def build_prescriptive_recommendations(df, trained):
    recommendations = []
    cluster_summary = trained["cluster_summary"]
    cluster_names = trained["cluster_names"]
    interest_by_cluster = trained["interest_by_cluster"]

    cluster_top_interest = (
        interest_by_cluster.pivot(index="cluster", columns=TARGET_CLASS, values="count").fillna(0)
    )
    if "Yes" in cluster_top_interest.columns:
        best_cluster = cluster_top_interest["Yes"].idxmax()
        best_name = cluster_names.get(best_cluster, f"Cluster {best_cluster}")
        recommendations.append(
            f"Primary focus segment: **{best_name}**. This cluster has the highest count of strong-interest respondents."
        )

    top_features = summarize_binary_features(df)
    top_features = top_features[top_features["group"] == "Preferred Features"].sort_values("selected_pct", ascending=False)
    if not top_features.empty:
        f1 = top_features.iloc[0]["feature"].replace("feature_", "").replace("_", " ").title()
        f2 = top_features.iloc[1]["feature"].replace("feature_", "").replace("_", " ").title()
        recommendations.append(
            f"Launch priority should emphasize **{f1}** and **{f2}**, because these are the most broadly selected features."
        )

    budget_avg = df[TARGET_REG].mean()
    if budget_avg < 1200:
        recommendations.append(
            "Pricing recommendation: start with a **freemium** model and low-cost premium tier, because the average monthly expenditure profile suggests high price sensitivity."
        )
    else:
        recommendations.append(
            "Pricing recommendation: test a **freemium + premium mentorship** structure, because the expenditure profile can support value-based upsells."
        )

    rules = trained["rules"]
    if isinstance(rules, pd.DataFrame) and not rules.empty:
        top_rule = rules.iloc[0]
        recommendations.append(
            f"Cross-sell rule to use in onboarding: if a user shows **{top_rule['antecedents']}**, recommend **{top_rule['consequents']}** because the rule has confidence {top_rule['confidence']:.2f} and lift {top_rule['lift']:.2f}."
        )

    recommendations.append(
        "Marketing recommendation: use classification probabilities to split new prospects into **High Intent**, **Nurture**, and **Low Priority** groups, and tailor messaging by cluster persona."
    )
    return recommendations

def safe_download_excel(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()

def display_metric_row(metrics):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    c2.metric("Precision (weighted)", f"{metrics['precision_weighted']:.3f}")
    c3.metric("Recall (weighted)", f"{metrics['recall_weighted']:.3f}")
    c4.metric("F1 score (weighted)", f"{metrics['f1_weighted']:.3f}")

def make_roc_figure(roc_records):
    fig = go.Figure()
    for rec in roc_records:
        fig.add_trace(
            go.Scatter(
                x=rec["fpr"], y=rec["tpr"], mode="lines",
                name=f"{rec['label']} (AUC={rec['auc']:.3f})"
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash"),
            name="Random baseline"
        )
    )
    fig.update_layout(
        title="One-vs-Rest ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=500,
    )
    return fig

def make_confusion_heatmap(cm, labels):
    z = cm
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=z,
            texttemplate="%{text}",
            hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        template="plotly_white",
        height=450,
    )
    return fig

def main():
    st.title("📊 Founder Analytics App for Career Launchpad")
    st.caption("Descriptive, diagnostic, predictive, and prescriptive analytics with future lead scoring.")

    with st.sidebar:
        st.header("Data Input")
        uploaded_main = st.file_uploader(
            "Upload main survey dataset (CSV or Excel). Leave blank to use the included dataset.",
            type=["csv", "xlsx", "xls"],
            key="main_file",
        )
        st.markdown(
            """
            **Expected target columns in training data**
            - `interest_in_app`
            - `monthly_expenditure_aed`

            **Future upload scoring**
            - The future dataset may omit target columns.
            """
        )

    if uploaded_main is not None:
        raw_df = read_dataset(uploaded_main)
        st.sidebar.success("Using uploaded main dataset.")
    else:
        raw_df = load_default_dataset()
        st.sidebar.info("Using bundled synthetic UAE survey dataset.")

    df = clean_base_dataframe(raw_df)
    trained = train_models(df)

    tabs = st.tabs([
        "Overview",
        "Descriptive",
        "Diagnostic",
        "Clustering",
        "Classification",
        "Regression",
        "Association Rules",
        "Prescriptive",
        "Future Upload Scoring",
    ])

    with tabs[0]:
        st.subheader("Business-ready overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Respondents", len(df))
        c2.metric("Columns", df.shape[1])
        c3.metric("Interested (Yes)", int((df[TARGET_CLASS] == "Yes").sum()))
        c4.metric("Average monthly expenditure", f"AED {df[TARGET_REG].mean():,.0f}")

        st.markdown(
            """
            This app is built to answer four founder questions:

            1. **Who should we target first?**
            2. **Which features matter most?**
            3. **How much can customers spend?**
            4. **How do we score future prospects automatically?**
            """
        )

        st.dataframe(df.head(10), use_container_width=True)

        missing_summary = df.isna().sum().reset_index()
        missing_summary.columns = ["column", "missing_count"]
        missing_summary["missing_pct"] = (missing_summary["missing_count"] / len(df) * 100).round(2)
        st.subheader("Missing value audit")
        st.dataframe(missing_summary.sort_values("missing_count", ascending=False), use_container_width=True)

    with tabs[1]:
        st.subheader("Descriptive analytics")

        interest_counts = df[TARGET_CLASS].value_counts(dropna=False).reset_index()
        interest_counts.columns = ["interest_in_app", "count"]
        fig_interest = px.bar(
            interest_counts,
            x="interest_in_app",
            y="count",
            title="Interest in app distribution",
            text="count",
        )
        st.plotly_chart(fig_interest, use_container_width=True)

        binary_summary = summarize_binary_features(df)

        group_choice = st.selectbox("Select binary feature group", list(BINARY_GROUPS.keys()))
        group_df = binary_summary[binary_summary["group"] == group_choice].copy()
        fig_group = px.bar(
            group_df.sort_values("selected_pct", ascending=True),
            x="selected_pct",
            y="feature",
            orientation="h",
            title=f"{group_choice}: selection percentage",
            text="selected_pct",
        )
        st.plotly_chart(fig_group, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if "field_of_study" in df.columns:
                field_counts = df["field_of_study"].value_counts().reset_index()
                field_counts.columns = ["field_of_study", "count"]
                st.plotly_chart(
                    px.pie(field_counts, names="field_of_study", values="count", title="Field of study mix"),
                    use_container_width=True,
                )
        with col2:
            st.plotly_chart(
                px.histogram(
                    df,
                    x=TARGET_REG,
                    nbins=35,
                    title="Monthly expenditure distribution",
                ),
                use_container_width=True,
            )

        st.subheader("Cross-view: interest by current status")
        cross_status = pd.crosstab(df["current_status"], df[TARGET_CLASS], normalize="index").mul(100).round(1)
        st.dataframe(cross_status, use_container_width=True)

    with tabs[2]:
        st.subheader("Diagnostic analytics")

        st.markdown("### Relationships between challenges and desired features")
        pair_rows = []
        for challenge in BINARY_GROUPS["Challenges"]:
            for feature in BINARY_GROUPS["Preferred Features"]:
                if challenge in df.columns and feature in df.columns:
                    selected = df[df[challenge] == 1][feature].mean() * 100
                    pair_rows.append({
                        "challenge": challenge,
                        "feature": feature,
                        "pct_feature_selected_given_challenge": round(selected, 2),
                    })
        pair_df = pd.DataFrame(pair_rows)
        heat_df = pair_df.pivot(index="challenge", columns="feature", values="pct_feature_selected_given_challenge")
        fig_heat = px.imshow(
            heat_df,
            aspect="auto",
            text_auto=True,
            title="Feature preference conditional on challenge selection (%)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("### Feature-level diagnostic importance")
        st.dataframe(trained["mutual_info"].head(15), use_container_width=True)

        st.markdown("### Interest by dissatisfaction with existing platforms")
        tmp = df.copy()
        tmp["satisfaction_band"] = pd.cut(
            tmp["platform_satisfaction"], bins=[0, 2, 3, 5], labels=["Low", "Medium", "High"], include_lowest=True
        )
        diag_interest = pd.crosstab(tmp["satisfaction_band"], tmp[TARGET_CLASS], normalize="index").mul(100).round(1)
        st.dataframe(diag_interest, use_container_width=True)

    with tabs[3]:
        st.subheader("Clustering and personas")

        sil_df = trained["silhouette_scores"]
        st.plotly_chart(
            px.line(sil_df, x="k", y="silhouette_score", markers=True, title="Silhouette scores by K"),
            use_container_width=True,
        )

        st.success(f"Selected optimal K for K-Means: {trained['optimal_k']}")

        cluster_plot = trained["cluster_plot"].copy()
        cluster_plot["cluster_name"] = cluster_plot["cluster"].astype(int).map(trained["cluster_names"])
        fig_cluster = px.scatter(
            cluster_plot,
            x="pc1",
            y="pc2",
            color="cluster_name",
            title="K-Means customer persona map (PCA projection)",
            hover_data=["latent_class"],
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("### Cluster profile summary")
        profile_df = trained["cluster_summary"].reset_index()
        profile_df["cluster_name"] = profile_df["cluster"].map(trained["cluster_names"])
        st.dataframe(profile_df, use_container_width=True)

        st.markdown("### Interest distribution by cluster")
        interest_cluster = trained["interest_by_cluster"].copy()
        interest_cluster["cluster_name"] = interest_cluster["cluster"].map(trained["cluster_names"])
        fig_ic = px.bar(
            interest_cluster,
            x="cluster_name",
            y="count",
            color=TARGET_CLASS,
            barmode="group",
            title="Interest classes by customer persona",
        )
        st.plotly_chart(fig_ic, use_container_width=True)

    with tabs[4]:
        st.subheader("Classification: predict interest in app")
        display_metric_row(trained["class_metrics"])

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(make_confusion_heatmap(
                trained["class_metrics"]["confusion_matrix"], trained["class_metrics"]["labels"]
            ), use_container_width=True)
        with col2:
            st.plotly_chart(make_roc_figure(trained["roc_records"]), use_container_width=True)

        st.markdown("### Top feature importance (permutation importance)")
        fi_top = trained["feature_importance"].head(20).sort_values("importance", ascending=True)
        fig_fi = px.bar(
            fi_top, x="importance", y="feature", orientation="h",
            title="Top 20 classification drivers"
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.dataframe(trained["feature_importance"].head(25), use_container_width=True)

    with tabs[5]:
        st.subheader("Regression: predict monthly expenditure")

        reg_metrics = trained["reg_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{reg_metrics['MAE']:.2f}")
        c2.metric("RMSE", f"{reg_metrics['RMSE']:.2f}")
        c3.metric("R²", f"{reg_metrics['R2']:.3f}")

        avp = reg_metrics["actual_vs_pred"].copy()
        fig_avp = px.scatter(
            avp, x="actual", y="predicted",
            trendline="ols",
            title="Actual vs predicted monthly expenditure"
        )
        st.plotly_chart(fig_avp, use_container_width=True)

        st.markdown(
            """
            **How to use this regression output**
            - High predicted expenditure + high interest → premium or mentorship plan
            - Moderate expenditure + moderate interest → freemium nurture
            - Low expenditure + high need → low-cost student plan
            """
        )

    with tabs[6]:
        st.subheader("Association rule mining")
        rules = trained["rules"]
        if isinstance(rules, pd.DataFrame) and not rules.empty:
            display_rules = rules[[
                "antecedents", "consequents", "support", "confidence", "lift"
            ]].head(25).copy()
            display_rules["support"] = display_rules["support"].round(3)
            display_rules["confidence"] = display_rules["confidence"].round(3)
            display_rules["lift"] = display_rules["lift"].round(3)

            st.dataframe(display_rules, use_container_width=True)

            fig_rules = px.scatter(
                display_rules,
                x="confidence",
                y="lift",
                size="support",
                hover_data=["antecedents", "consequents"],
                title="Association rules: confidence vs lift",
            )
            st.plotly_chart(fig_rules, use_container_width=True)
        else:
            st.warning("No association rules were found for the current support and confidence thresholds.")

    with tabs[7]:
        st.subheader("Prescriptive recommendations")
        recommendations = build_prescriptive_recommendations(df, trained)
        for i, rec in enumerate(recommendations, start=1):
            st.markdown(f"**{i}.** {rec}")

        st.markdown(
            """
            ### Suggested go-to-market logic
            - **High Intent**: Probability of Yes > 0.75 → push demo, early access, premium bundle
            - **Nurture**: Probability of Yes between 0.45 and 0.75 → send feature-led messaging
            - **Low Priority**: Probability of Yes < 0.45 → use awareness campaigns only

            ### Suggested launch wedge
            Start with **final-year students and freshers** who need:
            - resume support
            - internship/job alerts
            - mock interview preparation
            """
        )

    with tabs[8]:
        st.subheader("Upload future customers and predict inclination")
        st.markdown(
            """
            Upload a CSV or Excel file for new would-be customers.  
            The file can omit:
            - `interest_in_app`
            - `monthly_expenditure_aed`

            The app will score each record and generate:
            - predicted interest class
            - probability by class
            - predicted monthly expenditure
            - assigned customer persona
            - marketing action tag
            """
        )

        uploaded_future = st.file_uploader(
            "Upload new customer data",
            type=["csv", "xlsx", "xls"],
            key="future_file",
        )

        if uploaded_future is not None:
            future_df_raw = read_dataset(uploaded_future)
            future_df = clean_base_dataframe(future_df_raw)

            X_new_class = align_new_data_to_training(future_df, trained["class_training_columns"])
            class_model = trained["class_model"]
            prob = class_model.predict_proba(X_new_class)
            pred = class_model.predict(X_new_class)

            prob_df = pd.DataFrame(prob, columns=[f"prob_{c}" for c in class_model.classes_])

            X_new_reg = align_new_data_to_training(future_df, trained["reg_training_columns"])
            reg_pred = trained["reg_model"].predict(X_new_reg)

            X_new_cluster = align_new_data_to_training(future_df, trained["cluster_training_columns"])
            X_new_scaled = trained["cluster_scaler"].transform(X_new_cluster)
            cluster_pred = trained["kmeans"].predict(X_new_scaled)

            scored = future_df.copy()
            scored["predicted_interest"] = pred
            scored = pd.concat([scored.reset_index(drop=True), prob_df.reset_index(drop=True)], axis=1)
            scored["predicted_monthly_expenditure_aed"] = np.round(reg_pred, 2)
            scored["predicted_cluster"] = cluster_pred
            scored["predicted_cluster_name"] = scored["predicted_cluster"].map(trained["cluster_names"])

            yes_col = "prob_Yes" if "prob_Yes" in scored.columns else None
            maybe_col = "prob_Maybe" if "prob_Maybe" in scored.columns else None

            conditions = []
            for _, row in scored.iterrows():
                if yes_col and row.get(yes_col, 0) >= 0.75:
                    conditions.append("High Intent - Target Immediately")
                elif yes_col and row.get(yes_col, 0) >= 0.45:
                    conditions.append("Medium Intent - Nurture Campaign")
                else:
                    conditions.append("Low Priority - Awareness Only")
            scored["recommended_marketing_action"] = conditions

            st.success("Scoring completed.")
            st.dataframe(scored.head(25), use_container_width=True)

            export_bytes = safe_download_excel({
                "scored_customers": scored,
            })
            st.download_button(
                "Download scored results as Excel",
                data=export_bytes,
                file_name="scored_future_customers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.markdown("### Lead summary")
            if "predicted_interest" in scored.columns:
                lead_counts = scored["predicted_interest"].value_counts().reset_index()
                lead_counts.columns = ["predicted_interest", "count"]
                st.plotly_chart(
                    px.bar(lead_counts, x="predicted_interest", y="count", text="count", title="Predicted lead mix"),
                    use_container_width=True,
                )

    st.sidebar.download_button(
        "Download current cleaned dataset as Excel",
        data=safe_download_excel({"survey_data": df}),
        file_name="cleaned_training_dataset.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if __name__ == "__main__":
    main()
