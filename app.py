"""ML Studio — interactive, local no-code machine learning workspace."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

from mlstudio.insights import quality_report, generate_insights, score_guide

from mlstudio.core import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS, MAX_UPLOAD_BYTES,
    eligible_features, predictions_table, run_models, validate_data,
)

st.set_page_config(page_title="ML Studio | No-Code Machine Learning", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {--slate:#29384D;--steel:#53677F;--gold:#F4AD3D;--mist:#A4B6C9;}
[data-testid="stAppViewContainer"] {background: linear-gradient(155deg,#29384D 0%,#394B63 52%,#29384D 100%);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#27364A,#344861);border-right:1px solid rgba(240,243,247,.11);}
[data-testid="stHeader"] {background:rgba(41,56,77,.55);}
.block-container {max-width:1480px;padding-top:2rem;padding-bottom:4rem;}
h1,h2,h3 {letter-spacing:-.035em;}
h1 {font-weight:760!important;}
[data-testid="stMetric"] {background:linear-gradient(140deg,rgba(113,134,158,.28),rgba(41,56,77,.38));border:1px solid rgba(200,216,233,.18);border-radius:16px;padding:16px 20px;box-shadow:0 8px 22px rgba(12,23,40,.10);}
[data-testid="stMetricValue"] {font-size:1.7rem;letter-spacing:-.035em;}
div[data-testid="stPlotlyChart"],div[data-testid="stDataFrame"] {border:1px solid rgba(188,206,225,.15);border-radius:15px;overflow:hidden;}
[data-testid="stTabs"] button {border-radius:10px 10px 0 0;}
[data-testid="stExpander"] {border-radius:12px;}
.eyebrow{font-size:.75rem;letter-spacing:.14em;color:#F4AD3D;text-transform:uppercase;font-weight:750;}
.intro{color:#A4B6C9;font-size:1rem;margin-bottom:1.2rem;}
.hero {background:linear-gradient(120deg,rgba(113,134,158,.32),rgba(41,56,77,.24));border:1px solid rgba(200,216,233,.15);border-radius:20px;padding:23px 26px;margin:8px 0 18px;}
.hero-title {font-size:1.5rem;font-weight:760;color:#F0F3F7;}
.hero-copy{color:#A4B6C9;font-size:.94rem;margin-top:6px;}
.tagline {color:#F4AD3D;font-size:.73rem;text-transform:uppercase;letter-spacing:.1em;font-weight:700;}
</style>
""",unsafe_allow_html=True)

CHART_COLORS = ["#F4AD3D", "#A36CF1", "#35B99A", "#EF7883", "#75AEE7", "#E6C27D", "#9DD9C8"]


def chart_style(fig, height=380):
    fig.update_layout(
        height=height, margin=dict(l=18, r=18, t=35, b=18),
        font=dict(family="Arial", size=12, color="#F0F3F7"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="", hovermode="closest",
    )
    fig.update_xaxes(showgrid=False, color="#A4B6C9", linecolor="rgba(164,182,201,.22)")
    fig.update_yaxes(gridcolor="rgba(164,182,201,.12)", color="#A4B6C9")
    return fig


@st.cache_data(show_spinner=False)
def read_csv(contents: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(contents), low_memory=False)
    validate_data(df)
    return df


def dataset_fingerprint(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def signature(data_hash, target, features, problem, test_size, algorithms):
    payload = json.dumps([data_hash, target, features, problem, test_size, algorithms], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


st.markdown('<div class="eyebrow">DATA SCIENCE WORKSPACE</div>', unsafe_allow_html=True)
st.title("🧠 ML Studio Pro")
st.markdown('<div class="intro">Your complete no-code machine learning workspace · Explore · Experiment · Explain · Predict</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Workspace")
    upload = st.file_uploader("Upload a CSV dataset", type="csv", help="Maximum 20 MB; up to 100,000 rows")
    st.caption("Data is processed in your current Streamlit session. Don't upload confidential datasets to an untrusted deployment.")
    st.divider()
    st.markdown("**Try a bundled example**")
    demo_choice = st.selectbox("Example dataset", ["None", "Demo passenger survival", "Demo house prices"], label_visibility="collapsed")
    st.caption("Examples are synthetic, generated for demonstrations, not actual Titanic or housing data.")
    st.divider()
    st.markdown("**Workspace guide**")
    st.caption("Dashboard → Explore → Data Quality → Train → Evaluate → Predict. Your trained models remain available within this session.")

if upload is not None:
    if upload.size > MAX_UPLOAD_BYTES:
        st.error("The uploaded file exceeds the 20 MB limit.")
        st.stop()
    contents = upload.getvalue()
    label = upload.name
elif demo_choice != "None":
    example_path = "data/demo_passengers.csv" if demo_choice == "Demo passenger survival" else "data/demo_housing.csv"
    with open(example_path, "rb") as f:
        contents = f.read()
    label = demo_choice + " (synthetic)"
else:
    st.info("Upload a CSV from the sidebar or choose a bundled example to get started.")
    st.subheader("What you can build here")
    a, b, c = st.columns(3)
    a.info("📊 **Data Exploration**\n\nPreview, missing values, and interactive charts.")
    b.info("🧪 **Model Comparison**\n\nClassification and regression algorithms with evaluation.")
    c.info("🎯 **Prediction Studio**\n\nSingle and batch predictions with downloadable results.")
    st.stop()

try:
    data = read_csv(contents)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

hash_value = dataset_fingerprint(contents)
st.caption(f"Dataset: **{label}**")

dashboard, overview, quality, configure, evaluate, predict, experiments = st.tabs(["🏠 Dashboard", "📊 Explorer", "🛡️ Data Quality", "⚙️ Train", "📈 Evaluate", "🎯 Predict & Export", "🕓 Experiments"])

with dashboard:
    st.markdown('<div class="hero"><div class="tagline">Analytics overview</div><div class="hero-title">Make sense of your data. Build models with confidence.</div><div class="hero-copy">Your uploaded dataset is ready for exploration, diagnostics, training and predictions. Follow the tabs to complete the workflow.</div></div>', unsafe_allow_html=True)
    missing_total = int(data.isna().sum().sum())
    missing_pct = 100*missing_total/max(1,data.size)
    ka,kb,kc,kd = st.columns(4)
    ka.metric("Total records",f"{len(data):,}")
    kb.metric("Features / columns",len(data.columns))
    kc.metric("Missing cells",f"{missing_pct:.1f}%")
    kd.metric("Trained models", len(st.session_state.get("trained_result").models) if st.session_state.get("trained_result") is not None and st.session_state.get("trained_data_hash")==hash_value else 0)
    left,right = st.columns([1.2,1])
    with left:
        st.markdown("#### Dataset health")
        missing_counts = data.isna().sum().sort_values(ascending=False).head(12)
        if missing_counts.sum():
            health = pd.DataFrame({"Column":missing_counts.index,"Missing":missing_counts.values})
            fig=px.bar(health, x="Missing",y="Column",orientation="h",color="Missing",color_continuous_scale=["#53677F","#F4AD3D"],title="Most incomplete columns")
            fig.update_layout(coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
            st.plotly_chart(chart_style(fig,350),use_container_width=True)
        else:
            st.success("No missing data detected in this dataset.")
    with right:
        st.markdown("#### Dataset composition")
        counts = data.dtypes.astype(str)
        numeric_count = len(data.select_dtypes(include="number").columns)
        category_count = len(data.select_dtypes(include=["object","string","category"]).columns)
        other_count = len(data.columns)-numeric_count-category_count
        fig=px.pie(names=["Numerical","Categorical","Other"],values=[numeric_count,category_count,other_count],hole=.62,color_discrete_sequence=CHART_COLORS,title="Column types")
        st.plotly_chart(chart_style(fig,350),use_container_width=True)
    st.markdown("#### Start your workflow")
    a,b,c=st.columns(3)
    a.info("**01 · Explore**\n\nView distributions, relationships, statistics and correlations.")
    b.info("**02 · Train**\n\nChoose a target, features and multiple machine learning algorithms.")
    c.info("**03 · Predict**\n\nEvaluate performance and create predictions on new records.")
    st.caption("Tip: Use the Data Quality tab to identify issues before training. This application is intended for learning and experimentation.")

with overview:
    st.subheader("Dataset Overview")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(data):,}")
    b.metric("Columns", len(data.columns))
    c.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(data.duplicated().sum()):,}")
    st.dataframe(data.head(100), use_container_width=True, height=280)
    with st.expander("Column details & missing values", expanded=True):
        info = pd.DataFrame({
            "Column": data.columns, "Type": data.dtypes.astype(str).to_numpy(),
            "Missing": data.isna().sum().to_numpy(),
            "Missing %": (100 * data.isna().mean()).round(2).to_numpy(),
            "Unique values": data.nunique(dropna=True).to_numpy(),
        })
        st.dataframe(info, use_container_width=True, hide_index=True)
    st.subheader("Visual Explorer")
    view_col = st.selectbox("Column to visualize", data.columns.tolist(), key="explore_col")
    if pd.api.types.is_numeric_dtype(data[view_col]):
        fig = px.histogram(data, x=view_col, nbins=30, color_discrete_sequence=[CHART_COLORS[0]], title=f"Distribution of {view_col}")
    else:
        counts = data[view_col].fillna("(Missing)").astype(str).value_counts().head(15)
        fig = px.bar(x=counts.index, y=counts.to_numpy(), color=counts.index,
                     color_discrete_sequence=CHART_COLORS, title=f"Top categories in {view_col}",
                     labels={"x": view_col, "y": "Count"})
        fig.update_layout(showlegend=False)
    st.plotly_chart(chart_style(fig), use_container_width=True)
    numeric = data.select_dtypes(include="number")
    with st.expander("Numeric summary and correlations"):
        if not numeric.empty:
            st.dataframe(numeric.describe().T.round(3), use_container_width=True)
            if numeric.shape[1] >= 2:
                cor = numeric.corr(numeric_only=True).round(2)
                heat = px.imshow(cor, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", text_auto=True)
                st.plotly_chart(chart_style(heat, height=420), use_container_width=True)
        else:
            st.info("No numeric columns to summarize.")

    st.subheader("Interactive Relationship Explorer")
    numbers = data.select_dtypes(include="number").columns.tolist()
    if len(numbers)>=2:
        x_axis,y_axis=st.columns(2)
        with x_axis:
            xvar=st.selectbox("Horizontal axis",numbers,index=0,key="pair_x")
        with y_axis:
            yvar=st.selectbox("Vertical axis",numbers,index=min(1,len(numbers)-1),key="pair_y")
        display_points = data[[xvar,yvar]].dropna().sample(n=min(3000,len(data[[xvar,yvar]].dropna())),random_state=42)
        fig=px.scatter(display_points,x=xvar,y=yvar,opacity=.55,color_discrete_sequence=[CHART_COLORS[1]],title="Feature relationship · scatter plot")
        st.plotly_chart(chart_style(fig,380),use_container_width=True)
        st.caption("Up to 3,000 non-missing records are sampled for responsiveness.")
    with st.expander("Distribution laboratory · box plots and group frequencies"):
        plot_type=st.radio("Chart",["Box plot","Top category frequencies"],horizontal=True,key="lab_plot")
        if plot_type=="Box plot" and numbers:
            metric=st.selectbox("Numerical column",numbers,key="lab_numeric")
            bfig=px.box(data,y=metric,points=False,color_discrete_sequence=[CHART_COLORS[2]],title=f"Spread and outliers · {metric}")
            st.plotly_chart(chart_style(bfig),use_container_width=True)
        elif plot_type=="Top category frequencies":
            catcols=data.select_dtypes(include=["object","string","category"]).columns.tolist()
            if catcols:
                cat=st.selectbox("Categorical column",catcols,key="lab_category")
                counts=data[cat].fillna("Missing").astype(str).value_counts().head(12)
                fig=px.bar(x=counts.index,y=counts.values,color=counts.index,color_discrete_sequence=CHART_COLORS,labels={"x":cat,"y":"Count"})
                fig.update_layout(showlegend=False)
                st.plotly_chart(chart_style(fig),use_container_width=True)
            else:
                st.info("No categorical columns.")

with quality:
    st.subheader("Data Quality & Diagnostics")
    report=quality_report(data)
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Data completeness",f"{report['completeness']:.1f}%")
    q2.metric("Duplicate records",report['duplicates'])
    q3.metric("Constant columns",len(report['constant_columns']))
    q4.metric("Potential IDs",len(report['identifier_columns']))
    st.markdown("#### Column-level quality report")
    st.dataframe(report['columns'],use_container_width=True,hide_index=True)
    st.download_button("Download quality report CSV",report['columns'].to_csv(index=False).encode(),"data_quality_report.csv","text/csv")
    st.markdown("#### Recommended checks")
    for insight in generate_insights(data):
        st.info(insight)
    st.markdown("#### Dataset cleaning workspace")
    st.caption("The original dataset stays unchanged. Download a separate cleaned copy to inspect or use later.")
    drop_dupes=st.checkbox("Remove duplicate rows",value=True,key="clean_dupes")
    drop_empty=st.checkbox("Remove entirely empty columns",value=True,key="clean_empty")
    cleaned=data.copy()
    if drop_dupes:
        cleaned=cleaned.drop_duplicates()
    if drop_empty:
        cleaned=cleaned.dropna(axis=1,how="all")
    st.write(f"Result: {len(cleaned):,} rows × {cleaned.shape[1]} columns (original: {len(data):,} × {data.shape[1]})")
    st.download_button("Download cleaned CSV",cleaned.to_csv(index=False).encode(),"cleaned_dataset.csv","text/csv")

with configure:
    st.subheader("Prediction Setup")
    cols = data.columns.tolist()
    suggested_target = next((c for c in ("Survived", "Churn", "target", "SalePrice", "price", "Target") if c in cols), cols[-1])
    target = st.selectbox("Target variable (what you want to predict)", cols, index=cols.index(suggested_target))
    target_unique = data[target].nunique(dropna=True)
    suggested_problem = "Regression" if pd.api.types.is_numeric_dtype(data[target]) and target_unique > 15 else "Classification"
    problem = st.radio("Machine learning task", ["Classification", "Regression"], index=0 if suggested_problem == "Classification" else 1, horizontal=True)
    n, c, unsupported = eligible_features(data, target)
    recommended = [col for col in n+c if not (
        col.lower().endswith("id") or col.lower() in ("name", "ticket", "email", "customerid")
        or (col in c and data[col].nunique(dropna=True) > 30)
    )]
    if not recommended:
        recommended = (n+c)[:4]
    features = st.multiselect("Input features (X)", n+c, default=recommended[:12])
    if unsupported:
        st.warning("Excluded unsupported data types: " + ", ".join(unsupported))
    if features:
        n_selected = [col for col in features if col in n]
        c_selected = [col for col in features if col in c]
        st.caption(f"Numerical: {len(n_selected)} • Categorical: {len(c_selected)} • Target missing: {int(data[target].isna().sum())}")
        high_card = [x for x in c_selected if data[x].nunique(dropna=True) > 100]
        if high_card:
            st.warning("High-cardinality features need to be excluded: " + ", ".join(high_card))
        st.dataframe(data[features + [target]].head(), use_container_width=True)
    test_pct = st.slider("Test set percentage", 10, 40, 20, step=5)
    options = CLASSIFICATION_MODELS if problem == "Classification" else REGRESSION_MODELS
    defaults = ["Logistic Regression", "Random Forest"] if problem == "Classification" else ["Linear Regression", "Random Forest"]
    algorithms = st.multiselect("Models to train and compare", options, default=defaults)
    config_id = signature(hash_value, target, features, problem, test_pct, algorithms)
    if st.button("🚀 Train selected models", type="primary", use_container_width=True):
        try:
            with st.spinner("Training and evaluating models on the held-out test set..."):
                result = run_models(data, target, features, problem, test_pct / 100, algorithms)
            st.session_state["trained_result"] = result
            st.session_state["trained_data_hash"] = hash_value
            hist=st.session_state.setdefault("experiment_history",[])
            hist.insert(0,{"Timestamp UTC":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),"Dataset":label,"Target":target,"Problem":problem,"Models":", ".join(algorithms),"Feature count":len(features),"Test split":f"{test_pct}%","Top model":str(result.scores.iloc[0]["Model"]),"Top score":float(result.scores.iloc[0]["F1 (weighted)" if problem=="Classification" else "R²"])})
            if len(hist)>50: del hist[50:]
            st.session_state["trained_signature"] = config_id
            st.session_state["trained_details"] = (hash_value, target, tuple(features), problem, test_pct)
            st.success("Training completed! Open Evaluate or Predict & Export.")
        except Exception as exc:
            st.error(f"Training could not complete: {exc}")
    if st.session_state.get("trained_signature") == config_id:
        st.success("Models trained for these settings.")
        st.dataframe(st.session_state["trained_result"].scores.round(4), use_container_width=True, hide_index=True)
    elif "trained_result" in st.session_state:
        st.info("Your dataset or training settings changed. Retrain to evaluate or predict with the new configuration.")

# Tabs are executed on every rerun, so compute a stable model selection inside each tab.
valid_result = st.session_state.get("trained_signature") == config_id
result = st.session_state.get("trained_result") if valid_result else None

with evaluate:
    st.subheader("Model Comparison & Evaluation")
    if result is None:
        st.info("Choose settings in Configure & Train and click Train selected models.")
    else:
        scores = result.scores
        metric = "F1 (weighted)" if problem == "Classification" else "RMSE"
        st.caption("Classification metrics use support-weighted averaging across classes. Test set is held out from training; comparison is exploratory, not cross-validation.")
        st.dataframe(scores.style.format(precision=4), use_container_width=True, hide_index=True)
        fig = px.bar(scores, x="Model", y=metric, color="Model", color_discrete_sequence=CHART_COLORS, title=f"Model comparison · {metric}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_style(fig), use_container_width=True)
        name = st.selectbox("Inspect a model", scores["Model"].tolist(), key="inspect_model")
        y_true = result.y_test
        y_pred = result.predictions[name]
        with st.expander("How to interpret these metrics"):
            for note in score_guide(problem): st.write("• "+note)
        if problem == "Classification":
            labels = list(result.models[name].named_steps["model"].classes_)
            matrix = confusion_matrix(y_true, y_pred, labels=labels)
            chart = go.Figure(data=go.Heatmap(
                z=matrix, x=[f"Pred: {x}" for x in labels], y=[f"Actual: {x}" for x in labels],
                colorscale="Blues", text=matrix, texttemplate="%{text}", showscale=False,
            ))
            chart.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(chart_style(chart, 390), use_container_width=True)
            st.markdown("#### Class-by-class performance")
            summary = classification_report(y_true,y_pred,output_dict=True,zero_division=0)
            summary_df=pd.DataFrame(summary).T.round(3)
            st.dataframe(summary_df,use_container_width=True)
            st.download_button("Download classification report CSV",summary_df.to_csv().encode(),"classification_report.csv","text/csv")
            if len(labels)==2 and hasattr(result.models[name],"predict_proba"):
                try:
                    probabilities=result.models[name].predict_proba(result.x_test)[:,1]
                    binary=(pd.Series(y_true)==labels[1]).astype(int)
                    fpr,tpr,_=roc_curve(binary,probabilities)
                    auc=roc_auc_score(binary,probabilities)
                    roc=go.Figure()
                    roc.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",line=dict(color=CHART_COLORS[0],width=3),name=f"ROC · AUC {auc:.3f}"))
                    roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(color="#71869E",dash="dash"),name="Chance"))
                    roc.update_layout(title=f"ROC curve · AUC {auc:.3f}",xaxis_title="False positive rate",yaxis_title="True positive rate")
                    st.plotly_chart(chart_style(roc),use_container_width=True)
                except (ValueError,TypeError):
                    st.caption("ROC unavailable for this target/model combination.")
        else:
            sc = px.scatter(x=y_true, y=y_pred, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted", color_discrete_sequence=[CHART_COLORS[0]])
            limits = [min(float(np.min(y_true)), float(np.min(y_pred))), max(float(np.max(y_true)), float(np.max(y_pred)))]
            sc.add_trace(go.Scatter(x=limits, y=limits, mode="lines", line=dict(dash="dash", color=CHART_COLORS[1]), name="Ideal"))
            st.plotly_chart(chart_style(sc), use_container_width=True)
            residual = np.asarray(y_true) - np.asarray(y_pred)
            rf = px.histogram(x=residual, nbins=25, title="Prediction residuals (Actual − Predicted)", color_discrete_sequence=[CHART_COLORS[2]])
            st.plotly_chart(chart_style(rf), use_container_width=True)
        st.markdown("#### Feature influence")
        fitted_estimator=result.models[name].named_steps["model"]
        try:
            transformed_names=result.models[name].named_steps["preprocessor"].get_feature_names_out()
            if hasattr(fitted_estimator,"feature_importances_"):
                importance=fitted_estimator.feature_importances_
                method="Tree-based feature importance"
            elif hasattr(fitted_estimator,"coef_"):
                coefs=np.asarray(fitted_estimator.coef_)
                importance=np.mean(np.abs(coefs),axis=0) if coefs.ndim==2 else np.abs(coefs)
                method="Absolute model coefficients (affected by feature scaling)"
            else:
                importance=None
                method=""
            if importance is not None and len(importance)==len(transformed_names):
                importance_df=pd.DataFrame({"Feature":transformed_names,"Importance":importance}).sort_values("Importance",ascending=False).head(15)
                chart=px.bar(importance_df.sort_values("Importance"),x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#53677F","#F4AD3D"])
                chart.update_layout(coloraxis_showscale=False,title=method)
                st.plotly_chart(chart_style(chart,420),use_container_width=True)
                st.caption("Associations learned by this model, not causal explanations. Importance cannot be compared directly between different model families.")
            else:
                st.info("This algorithm does not expose built-in feature importance. Try Decision Tree or Random Forest.")
        except (AttributeError,ValueError,TypeError):
            st.caption("Feature influence is unavailable for this model configuration.")
        comparison = pd.DataFrame({"Actual": y_true.to_numpy(), "Predicted": y_pred}, index=y_true.index)
        st.dataframe(comparison.head(100), use_container_width=True)
        st.download_button("Download model comparison (CSV)", scores.to_csv(index=False).encode("utf-8"), "model_comparison.csv", "text/csv")

with predict:
    st.subheader("Prediction Studio")
    if result is None:
        st.info("Train models with the current settings before generating predictions.")
    else:
        name = st.selectbox("Model for predictions", result.scores["Model"].tolist(), key="prediction_model")
        pipeline = result.models[name]
        input_tab, batch_tab, export_tab = st.tabs(["Single prediction", "Batch CSV prediction", "Export model"])
        with input_tab:
            st.caption("Values are processed with the same fitted transformations used during training.")
            single = {}
            with st.form("single_prediction_form"):
                ui_cols = st.columns(2)
                for idx, field in enumerate(result.feature_columns):
                    with ui_cols[idx % 2]:
                        if field in result.numeric_columns:
                            valid = pd.to_numeric(data[field], errors="coerce").dropna()
                            default_val = float(valid.median()) if len(valid) else 0.0
                            single[field] = st.number_input(field, value=default_val)
                        else:
                            options_cat = sorted(data[field].dropna().astype(str).unique().tolist())
                            single[field] = st.selectbox(field, options_cat, key=f"pred_{field}")
                submitted = st.form_submit_button("Generate prediction", type="primary")
            if submitted:
                try:
                    prediction = predictions_table(pipeline, pd.DataFrame([single]), result.feature_columns)
                    st.success(f"Predicted outcome: {prediction['Prediction'].iloc[0]}")
                    st.dataframe(prediction, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        with batch_tab:
            st.markdown("Upload a CSV containing the same feature column names. Additional columns are preserved in the output.")
            template = data[result.feature_columns].head(5).to_csv(index=False).encode("utf-8")
            st.download_button("Download example input template", template, "prediction_template.csv", "text/csv")
            batch_file = st.file_uploader("New data (CSV)", type="csv", key="batch_file")
            if batch_file is not None:
                if batch_file.size > MAX_UPLOAD_BYTES:
                    st.error("Batch file exceeds 20 MB.")
                else:
                    try:
                        batch = read_csv(batch_file.getvalue())
                        if st.button("Predict all rows", type="primary"):
                            output = predictions_table(pipeline, batch, result.feature_columns)
                            st.session_state["batch_output"] = output
                            st.session_state["batch_signature"] = (config_id, name, hashlib.sha256(batch_file.getvalue()).hexdigest())
                    except Exception as exc:
                        st.error(f"Could not process batch CSV: {exc}")
                    if st.session_state.get("batch_signature") == (config_id, name, hashlib.sha256(batch_file.getvalue()).hexdigest()):
                        output = st.session_state["batch_output"]
                        st.success(f"Generated {len(output):,} predictions.")
                        st.dataframe(output.head(100), use_container_width=True)
                        st.download_button("Download predictions (CSV)", output.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")
        with export_tab:
            st.warning("Only load saved model files you trust. joblib/pickle files can execute code when loaded. Use a matching Python and package environment.")
            buffer = io.BytesIO()
            joblib.dump(pipeline, buffer)
            st.download_button("Download trained model (.joblib)", buffer.getvalue(), f"{name.lower().replace(' ', '_')}_model.joblib", "application/octet-stream")
            metadata = {"model": name, "problem": problem, "features": result.feature_columns, "target": target,
                        "training_export_utc": datetime.now(timezone.utc).isoformat(), "note": "Training data is not included."}
            st.download_button("Download model metadata (JSON)", json.dumps(metadata, indent=2), "model_metadata.json", "application/json")

with experiments:
    st.subheader("Experiment History")
    st.caption("Your recent training runs are recorded in this browser session. The history resets when the session ends; export it to keep a copy.")
    history=st.session_state.get("experiment_history",[])
    if history:
        experiments_df=pd.DataFrame(history)
        st.dataframe(experiments_df,use_container_width=True,hide_index=True)
        st.download_button("Download experiment log CSV",experiments_df.to_csv(index=False).encode(),"experiment_history.csv","text/csv")
        if st.button("Clear session history"):
            st.session_state["experiment_history"]=[]
            st.rerun()
    else:
        st.info("No experiments recorded yet. Train a model to create your first entry.")
    st.markdown("#### Helpful experiment guidelines")
    st.write("Compare algorithms on the same split. Do not repeatedly choose features based on the held-out test score; that can overestimate performance.")
    st.write("For serious use, keep an independent final test set and validate on representative future data.")

st.divider()
st.caption("ML Studio • A learning and portfolio project. Metrics are measured on a held-out test set; avoid using the tool for high-stakes decisions without independent validation.")
