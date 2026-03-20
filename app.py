import io
import json
import time
import pickle
import requests
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder

# Project modules (ensure these exist)
from modules.eda import EDAAnalyzer
from modules.data_cleaner import DataCleaner



# ---------------- PAGE CONFIG (call once) ---------------- #
st.set_page_config(page_title="DataPilot AI Studio", page_icon="assets/M2 logo.png", layout="wide")

# ---------------- LOTTIE HELPER ---------------- #
def st_lottie(lottie_source, height=360, key=None):
    if isinstance(lottie_source, str):
        url = lottie_source
        html = f"""
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <lottie-player src="{url}" background="transparent" speed="1"
            style="width:100%; height:{height}px;" loop autoplay></lottie-player>
        """
        components.html(html, height=height+20)
        return
    try:
        animation_data = json.dumps(lottie_source)
        html = f"""
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <div id="lottie-{key or 'anim'}"></div>
        <script>
          const c = document.getElementById("lottie-{key or 'anim'}");
          const p = document.createElement("lottie-player");
          p.background="transparent"; p.speed=1; p.style.width="100%"; p.style.height="{height}px"; p.loop=true; p.autoplay=true;
          c.appendChild(p);
          p.load({animation_data});
        </script>
        """
        components.html(html, height=height+20)
    except Exception:
        st.warning("⚠️ Could not render Lottie animation.")

def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ---------------- SESSION ---------------- #
if "entered_app" not in st.session_state:
    st.session_state.entered_app = False

for k in ["data", "cleaned_data", "pipeline_obj", "quality_score", "quality_label", "insights_cache"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ---------------- WELCOME ---------------- #
if not st.session_state.entered_app:

    # Full center wrapper
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

    # Animation
    st_lottie(
        "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json",
        height=300,
        key="welcome_animation"
    )

    # Title (centered)
    st.markdown(
        "<h1 style='color:white; text-align:center;'>👋 Welcome to DataPilot AI Studio</h1>",
        unsafe_allow_html=True
    )

    # Subtitle (centered)
    st.markdown(
        """
        <p style='color:#d0d0d0; font-size:18px; text-align:center;'>
        Upload a CSV/Excel file and get instant insights,<br>
        one-click perform smart cleaning, visualizations,<br>
        generate model predictions<br>
        and download a complete PDF report of your analysis.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Centered, small-width button ----
    col1, col2, col3 = st.columns([3, 1, 3])  # Middle column narrower = smaller button
    with col2:
        if st.button("🚀 Enter App", use_container_width=True):
            st.session_state.entered_app = True
            st.rerun()

    st.stop()


# ---------------- APP LAYOUT ---------------- #
st.markdown("<h1 style='text-align:center;'>🧠 DataPilot AI Studio</h1>", unsafe_allow_html=True)
st.sidebar.title("📍 Navigation")
selected_tab = st.sidebar.radio("Choose your step:", [
    "📤 Data Upload",
    "⚡ Smart Cleaning",
    "🧹 Data Processing",
    "📊 Visualization",
    "📈 Model Prediction",
    "📄 Generate PDF",
    "🔎 Explain My Dataset",
    "ℹ️ About"
])

if st.sidebar.button("🏠 Restart App"):
    st.session_state.entered_app = False
    st.rerun()

st.markdown("---")
status_cols = st.columns(4)
status_cols[0].markdown("📥 **Data Loaded:** " + ("✅" if st.session_state.data is not None else "❌"))
status_cols[1].markdown("🧹 **Data Processed:** " + ("✅" if st.session_state.cleaned_data is not None else "⚠️"))
status_cols[2].markdown("🔎 **Insights Ready:** " + ("✅" if st.session_state.cleaned_data is not None or st.session_state.data is not None else "⚠️"))
st.markdown("---")

# ---------------- Utility helpers ---------------- #
def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        s = str(c).strip()
        s = s.replace(" ", "_").replace("-", "_")
        s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
        s = s.lower()
        new_cols.append(s)
    df.columns = new_cols
    return df

def auto_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "missing")
    return df

def convert_datatypes_auto(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception:
                pass
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().sum() / max(1, len(parsed)) > 0.6:
                    df[col] = parsed
            except Exception:
                pass
    return df

def drop_useless_columns(df: pd.DataFrame, missing_thresh=0.95) -> pd.DataFrame:
    drop_cols = []
    for col in df.columns:
        try:
            if df[col].isnull().mean() > missing_thresh:
                drop_cols.append(col)
            elif df[col].nunique() <= 1:
                drop_cols.append(col)
        except Exception:
            pass
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df

def auto_handle_outliers(df: pd.DataFrame, action="remove", iqr_multiplier=1.5):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            if action == "remove":
                mask &= (df[col] >= lower) & (df[col] <= upper)
            else:
                df[col] = np.where(df[col] < lower, lower, df[col])
                df[col] = np.where(df[col] > upper, upper, df[col])
        except Exception:
            pass
    if action == "remove":
        df = df.loc[mask]
    return df

def scale_numeric(df: pd.DataFrame, method="standard"):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return df, None
    if method == "standard":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, scaler
    elif method == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df, scaler
    else:
        return df, None

def compute_quality(df: pd.DataFrame):
    rows = max(1, df.shape[0]); cols = max(1, df.shape[1])
    total = rows * cols
    missing_cells = int(df.isnull().sum().sum())
    missing_pct = (missing_cells / total) * 100.0
    duplicates = int(df.duplicated().sum()); dup_ratio = duplicates / rows
    zero_var = int((df.nunique() <= 1).sum()); zratio = zero_var / cols
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    high_card = sum(1 for c in cat_cols if rows and df[c].nunique() / rows > 0.9)
    high_card_ratio = high_card / (len(cat_cols) if len(cat_cols) else 1)
    score = 100.0
    score -= missing_pct * 0.6
    score -= dup_ratio * 40.0
    score -= zratio * 25.0
    score -= high_card_ratio * 15.0
    if rows < 10: score -= 10.0
    elif rows < 50: score -= 5.0
    score = max(0.0, min(100.0, round(score,2)))
    label = "Excellent ✅" if score>=85 else ("Good 👍" if score>=70 else ("Fair ⚠️" if score>=50 else "Poor ❌"))
    return dict(score=score, label=label, missing_pct=round(missing_pct,2), duplicates=duplicates, zero_var=zero_var, high_card=high_card, mem_mb=round(df.memory_usage(deep=True).sum()/1024**2,2))

# ---------------- Data Upload ---------------- #
if selected_tab == "📤 Data Upload":
    st.subheader("📤 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        eda = EDAAnalyzer()
        df = eda.read_file(uploaded_file)
        st.session_state.data = df
        st.session_state.cleaned_data = None
        st.success("✅ File loaded successfully!")
        st.dataframe(df.head())

# ---------------- One-Click Smart Cleaning ---------------- #
elif selected_tab == "⚡ Smart Cleaning":
    st.subheader("⚡ Smart Cleaning")
    if st.session_state.data is None:
        st.warning("Please upload a dataset first.")
    else:
        st.markdown("Use these quick actions to auto-clean your dataset. Each button performs a safe, transparent operation.")
        df0 = st.session_state.data.copy()

        col1, col2, col3 = st.columns(3)
        if col1.button("Auto Clean Dataset (recommended)"):
            try:
                df = df0.copy()
                df = fix_column_names(df)
                df = auto_fill_missing(df)
                df = convert_datatypes_auto(df)
                df = drop_useless_columns(df)
                df = auto_handle_outliers(df, action="cap")
                df, scaler_obj = scale_numeric(df, method="standard")
                st.session_state.cleaned_data = df
                st.success("Auto clean finished and saved to session.cleaned_data")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Auto clean failed: {e}")

        if col1.button("Remove duplicates"):
            df = df0.drop_duplicates()
            st.session_state.cleaned_data = df
            st.success("Duplicates removed.")
            st.dataframe(df.head())

        if col1.button("Fix column names"):
            df = fix_column_names(df0)
            st.session_state.data = df
            st.success("Column names normalized (snake_case).")
            st.dataframe(df.head())

        if col2.button("Auto-fill missing values"):
            df = auto_fill_missing(df0.copy())
            st.session_state.cleaned_data = df
            st.success("Missing values auto-filled (mean/mode).")
            st.dataframe(df.head())

        if col2.button("Convert datatypes (auto)"):
            df = convert_datatypes_auto(df0.copy())
            st.session_state.cleaned_data = df
            st.success("Datatypes converted where possible (numbers/dates).")
            st.dataframe(df.head())

        if col2.button("Drop useless columns"):
            df = drop_useless_columns(df0.copy())
            st.session_state.cleaned_data = df
            st.success("Dropped columns with >95% missing or zero variance.")
            st.dataframe(df.head())

        if col3.button("Auto-handle outliers (cap)"):
            df = auto_handle_outliers(df0.copy(), action="cap")
            st.session_state.cleaned_data = df
            st.success("Outliers capped (IQR).")
            st.dataframe(df.head())

        if col3.button("Auto-handle outliers (remove)"):
            df = auto_handle_outliers(df0.copy(), action="remove")
            st.session_state.cleaned_data = df
            st.success("Outlier rows removed (IQR).")
            st.dataframe(df.head())

        if col3.button("Standardize numeric columns"):
            df, scaler_obj = scale_numeric(df0.copy(), method="standard")
            st.session_state.cleaned_data = df
            st.session_state.pipeline_obj = st.session_state.get("pipeline_obj", {})
            st.success("Numeric columns standardized (zero mean, unit variance).")
            st.dataframe(df.head())

        if st.button("Normalize numeric columns (MinMax)"):
            df, scaler_obj = scale_numeric(df0.copy(), method="minmax")
            st.session_state.cleaned_data = df
            st.success("Numeric columns normalized to [0,1].")
            st.dataframe(df.head())

        st.markdown("---")
        st.info("Tip: after using One-Click actions, open **Data Processing** to apply more controlled cleaning and save pipeline.")

# ---------------- Data Processing ---------------- #
elif selected_tab == "🧹 Data Processing":
    st.subheader("🧹 Auto Data Cleaning (advanced)")
    if st.session_state.cleaned_data is None and st.session_state.data is None:
        st.warning("Please upload a dataset (Data Upload) or use One-Click Smart Cleaning first.")
    else:
        df_base = st.session_state.cleaned_data.copy() if st.session_state.cleaned_data is not None else st.session_state.data.copy()
        cleaner = DataCleaner()
        drop_duplicates = st.checkbox("🗑️ Remove Duplicates", True)
        fill_missing = st.checkbox("💧 Fill Missing Values", True)
        convert_types = st.checkbox("🕒 Convert Text to Date/Time", True)

        if st.button("🚀 Clean Data"):
            try:
                cleaned_df, orig_shape, new_shape = cleaner.clean(df_base.copy(), drop_duplicates, fill_missing, convert_types)
                st.session_state.cleaned_data = cleaned_df
                st.success("✅ Data cleaned successfully!")
                st.write(f"🔹 Original shape: {orig_shape}")
                st.write(f"🔹 New shape: {new_shape}")
                st.dataframe(cleaned_df.head())

                csv = cleaned_df.to_csv(index=False, encoding='utf-8', errors='replace')
                st.download_button("⬇️ Download Cleaned Data", data=csv, file_name="cleaned_data.csv", mime="text/csv")

                quality = compute_quality(cleaned_df)
                st.markdown("## 📊 Exploratory Data Analysis — Auto Summary")
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                col1.metric("📏 Rows", f"{cleaned_df.shape[0]:,}")
                col2.metric("📐 Columns", cleaned_df.shape[1])
                col3.metric("🧠 Memory (MB)", f"{quality['mem_mb']:.2f}")
                col4.metric("🧼 Missing (%)", f"{quality['missing_pct']:.2f}%")
                col5.metric("♻️ Duplicates", quality['duplicates'])
                col6.metric("🧱 Zero Var Cols", quality['zero_var'])
                col7.metric("📊 Quality Score", f"{quality['score']}/100 — {quality['label']}")

                numerics = cleaned_df.select_dtypes(include="number").columns.tolist()
                categoricals = cleaned_df.select_dtypes(exclude="number").columns.tolist()
                st.markdown(f"🔢 **Numeric Columns ({len(numerics)}):** `{', '.join(numerics[:8]) + ('...' if len(numerics) > 8 else '')}`")
                st.markdown(f"🔠 **Categorical Columns ({len(categoricals)}):** `{', '.join(categoricals[:8]) + ('...' if len(categoricals) > 8 else '')}`")

                with st.expander("📈 Show Descriptive Statistics"):
                    st.dataframe(cleaned_df.describe(include="all").T)
            except Exception as e:
                st.error(f"❌ Data processing failed: {e}")

# ---------------- Visualization  ---------------- #
elif selected_tab == "📊 Visualization":
    st.subheader("📊 Explore Visualizations")
    if st.session_state.cleaned_data is None:
        st.warning("🧹 Please clean your data first.")
    else:
        df = st.session_state.cleaned_data
        eda = EDAAnalyzer()

        chart_type = st.selectbox("🎛️ Select a chart type:", [
            "Histogram", "Box Plot", "Violin Plot", "Density Plot",
            "Scatter Plot", "Correlation Heatmap", "Pair Plot",
            "Bar Plot", "Pie Chart"
        ])

        if chart_type == "Scatter Plot":
            x_axis = st.selectbox("📊 Select X-axis", df.columns.tolist(), key="scatter_x_axis")
            y_axis = st.selectbox("📈 Select Y-axis", df.columns.tolist(), key="scatter_y_axis")

            if x_axis and y_axis:
                if st.button("📈 Generate Scatter Plot"):
                    try:
                        eda.plot_scatterplots(df, x_col=x_axis, y_col=y_axis)
                    except Exception as e:
                        st.error(f"❌ Error generating scatter plot: {e}")
            else:
                st.info("ℹ️ Please select both X and Y axes.")

        else:
            if st.button(f"📈 Generate {chart_type}"):
                try:
                    chart_map = {
                        "Histogram": eda.plot_histograms,
                        "Box Plot": eda.plot_boxplots,
                        "Violin Plot": eda.plot_violinplots,
                        "Density Plot": eda.plot_density,
                        "Correlation Heatmap": eda.plot_heatmap,
                        "Pair Plot": eda.plot_pairplot,
                        "Bar Plot": eda.plot_barplots,
                        "Pie Chart": eda.plot_piecharts,
                    }
                    chart_map[chart_type](df)
                except Exception as e:
                    st.error(f"❌ Error generating {chart_type}: {e}")


# ---------------- Model Prediction ---------------- #
elif selected_tab == "📈 Model Prediction":

    st.subheader("📈 Model Prediction")

    if "cleaned_data" not in st.session_state:
        st.warning("Please clean the dataset first.")
    else:
        df = st.session_state.cleaned_data.copy()

        if df.shape[0] < 5:
            st.error("Dataset too small for prediction.")
            st.stop()

        st.write("Dataset Shape:", df.shape)
        st.dataframe(df.head())

        # Select Target Column
        target = st.selectbox("Select Target Column", df.columns)

        if target:

            # Remove rows where target is missing
            df = df[df[target].notna()]

            # Fill numeric missing values
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # Fill remaining categorical missing values
            df = df.fillna("Unknown")

            # Define X and y
            X = df.drop(columns=[target])
            y = df[target]

            # Keep only numeric features for model
            X = X.select_dtypes(include=['int64', 'float64'])

            if X.shape[1] == 0:
                st.error("No numeric features available for training.")
                st.stop()

            if X.shape[0] < 5:
                st.error("Not enough valid rows to train model.")
                st.stop()

            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.metrics import r2_score, accuracy_score
            from sklearn.preprocessing import LabelEncoder

            # Encode categorical target
            if y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Decide model type
            if len(set(y)) > 10:
                # Regression
                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

                st.success("Regression Model Trained Successfully")
                st.write("R² Score:", round(score, 3))

                st.session_state.model_score = score
                st.session_state.model_type = "Linear Regression"
                st.session_state.target_column = target

            else:
                # Classification
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)

                st.success("Classification Model Trained Successfully")
                st.write("Accuracy:", round(score, 3))

                st.session_state.model_score = score
                st.session_state.model_type = "Logistic Regression"
                st.session_state.target_column = target

# ---------------- Generate PDF Report ---------------- #
elif selected_tab == "📄 Generate PDF":

    st.subheader("📄 Generate PDF Report")

    if "model_score" not in st.session_state:
        st.warning("Please train model first before generating report.")
    else:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph
        from reportlab.platypus import Spacer
        from reportlab.platypus import Table

        file_path = "Model_Report.pdf"
        doc = SimpleDocTemplate(file_path, pagesize=A4)

        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("<b>AI Data Analysis Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph(f"Dataset Shape: {st.session_state.cleaned_data.shape}", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(f"Target Column: {st.session_state.target_column}", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(f"Model Type: {st.session_state.model_type}", styles["Normal"]))
        elements.append(Spacer(1, 0.2 * inch))

        elements.append(Paragraph(f"Model Performance Score: {round(st.session_state.model_score,3)}", styles["Normal"]))
        elements.append(Spacer(1, 0.3 * inch))

        elements.append(Paragraph("This report is automatically generated by the AI Data Studio application.", styles["Normal"]))

        doc.build(elements)

        with open(file_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="AI_Model_Report.pdf",
                mime="application/pdf"
            )

        st.success("PDF Generated Successfully!")


# ---------------- Explain My Dataset (Attractive Local Summary) ---------------- #
elif selected_tab == "🔎 Explain My Dataset":
    st.subheader("🔎 Explain My Dataset with AI")

    if st.session_state.cleaned_data is None and st.session_state.data is None:
        st.warning("Please upload or clean a dataset first. 📤")
    else:
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data

        if st.button("✨ Summary"):
            try:
                import numpy as np

                rows, cols = df.shape
                mem_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 3)
                total_cells = max(1, rows * cols)
                missing_cells = int(df.isnull().sum().sum())
                missing_pct = round(missing_cells / total_cells * 100, 2)
                dup_count = int(df.duplicated().sum())

                # Quality score
                def quality_score(df):
                    rows, cols = df.shape
                    total = max(1, rows * cols)
                    miss = df.isnull().sum().sum() / total
                    dup = df.duplicated().sum() / max(1, rows)
                    zero_var = (df.nunique() <= 1).sum() / max(1, cols)
                    score = 100 - (miss * 50) - (dup * 20) - (zero_var * 30)
                    if rows < 20:
                        score -= 5
                    return max(0, round(score, 2))
                qscore = quality_score(df)

                # Column overview
                col_rows = []
                for c in df.columns:
                    ser = df[c]
                    dtype = str(ser.dtype)
                    missing = int(ser.isnull().sum())
                    missing_pct_col = round(missing / max(1, rows) * 100, 2)
                    nunique = ser.nunique(dropna=True)
                    sample_vals = ser.dropna().astype(str).unique()[:3].tolist()
                    if pd.api.types.is_numeric_dtype(ser):
                        desc = ser.describe()
                        mean = round(desc.get("mean", np.nan), 4) if not pd.isna(desc.get("mean", np.nan)) else ""
                        std = round(desc.get("std", np.nan), 4) if not pd.isna(desc.get("std", np.nan)) else ""
                        p50 = round(ser.quantile(0.5), 4) if ser.count() else ""
                        top_values = f"mean={mean}, std={std}, median={p50}"
                    else:
                        vc = ser.fillna("<<MISSING>>").value_counts().head(5)
                        top_values = "; ".join([f"{idx} ({cnt})" for idx, cnt in zip(vc.index.astype(str), vc.values)])
                    col_rows.append({
                        "column": c,
                        "dtype": dtype,
                        "missing": missing,
                        "missing_%": missing_pct_col,
                        "unique": nunique,
                        "top_values_or_stats": top_values,
                        "examples": ", ".join(sample_vals[:3])
                    })
                col_summary_df = pd.DataFrame(col_rows).sort_values(by="missing_%", ascending=False)

                # Numeric correlations
                num = df.select_dtypes(include="number").copy()
                corr_pairs = []
                if num.shape[1] >= 2 and num.dropna().shape[0] > 0:
                    corr = num.corr().abs()
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            a = corr.columns[i]; b = corr.columns[j]; v = corr.iloc[i, j]
                            if v >= 0.5:
                                corr_pairs.append((a, b, round(v, 3)))
                    corr_pairs = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:10]

                # Cardinality and skew
                cat = df.select_dtypes(exclude="number").copy()
                high_card_cols = [c for c in cat.columns if df[c].nunique(dropna=True) > max(20, 0.1 * rows)]
                skewed_cols = []
                for c in num.columns:
                    try:
                        s = num[c].dropna()
                        if s.shape[0] > 0 and abs(s.skew()) > 1:
                            skewed_cols.append((c, round(s.skew(), 3)))
                    except Exception:
                        pass
                skewed_cols = sorted(skewed_cols, key=lambda x: abs(x[1]), reverse=True)

                # Executive summary lines
                exec_lines = []
                exec_lines.append(f"📌 **Rows:** **{rows:,}** &nbsp; • &nbsp; **Columns:** **{cols}**")
                exec_lines.append(f"💾 **Memory:** **{mem_mb} MB** &nbsp; • &nbsp; **Quality score:** **{qscore}/100**")
                exec_lines.append(f"⚠️ **Missing:** **{missing_cells}** cells ({missing_pct}%) &nbsp; • &nbsp; **Duplicates:** **{dup_count}**")
                if corr_pairs:
                    exec_lines.append(f"🔗 Top correlations: " + ", ".join([f"{a}-{b}={v}" for a,b,v in corr_pairs[:3]]))
                if high_card_cols:
                    exec_lines.append(f"🏷️ High-cardinality: {', '.join(high_card_cols[:4])}")

                # Save to state
                st.session_state["attractive_summary_text"] = "\n".join(exec_lines)
                st.session_state["col_summary_df"] = col_summary_df
                st.session_state["corr_pairs"] = corr_pairs
                st.session_state["skewed_cols"] = skewed_cols

                st.success("✨ Attractive summary generated!")
            except Exception as e:
                st.error(f"Failed to generate summary: {e}")

        # ------- Display attractive UI ------- #
        if st.session_state.get("attractive_summary_text"):
            st.markdown("## 🧾 Executive Snapshot")
            st.markdown(st.session_state["attractive_summary_text"], unsafe_allow_html=True)

            # metrics row
            m1, m2, m3, m4 = st.columns(4)
            q = compute_quality(df) if 'compute_quality' in globals() else {"score": None}
            m1.metric("📏 Rows", f"{df.shape[0]:,}")
            m2.metric("📐 Columns", df.shape[1])
            m3.metric("📊 Quality Score", f"{q.get('score','-')}/100" if isinstance(q, dict) else "-")
            m4.metric("💾 Memory (MB)", round(df.memory_usage(deep=True).sum() / 1024**2, 2))

            # Missing values bar chart (top 8)
            st.markdown("### 🔍 Missing Values (Top columns)")
            missing_series = df.isnull().mean().sort_values(ascending=False).head(8) * 100
            if missing_series.sum() > 0:
                fig_m, ax_m = plt.subplots(figsize=(8, 3))
                sns.barplot(x=missing_series.values, y=missing_series.index, orient="h", ax=ax_m)
                ax_m.set_xlabel("Missing %")
                st.pyplot(fig_m)
            else:
                st.success("No missing values detected 🎉")

            # Column summary table (first 20 rows)
            st.markdown("### 🧩 Column Overview (top issues first)")
            st.dataframe(st.session_state["col_summary_df"].head(20))

            # Strong correlations
            if st.session_state.get("corr_pairs"):
                st.markdown("### 🔗 Strong Numeric Correlations (abs ≥ 0.5)")
                corr_df = pd.DataFrame(st.session_state["corr_pairs"], columns=["col1","col2","corr"])
                st.table(corr_df.head(10))

            # Skewed numeric columns
            if st.session_state.get("skewed_cols"):
                st.markdown("### 📈 Highly Skewed Numeric Columns")
                st.write(", ".join([f"{c} (skew={s})" for c,s in st.session_state["skewed_cols"][:8]]))

            # Show sample rows
            st.markdown("### 🧾 Sample Rows")
            st.dataframe(df.head(6))

            # Actionable checklist with emoji badges
            st.markdown("### ✅ Actionable Checklist (prioritized)")
            # --- SAFE: Recalculate missing_pct here ---
            rows, cols = df.shape
            total_cells = max(1, rows * cols)
            missing_cells = int(df.isnull().sum().sum())
            missing_pct = round(missing_cells / total_cells * 100, 2)

            checklist = []
            # Missing values
            if missing_pct > 5:
                 checklist.append("🔸 Handle missing values: impute (mean/median/mode) or drop low-utility columns.")
            # Duplicate rows
            if df.duplicated().sum() > 0:
                checklist.append("🔸 Remove or investigate duplicate rows.")
            # Skewed columns
            if st.session_state.get("skewed_cols"):
                checklist.append("🔸 Transform skewed numeric columns (log / box-cox).")
            # High-cardinality columns (SAFE check)
            col_summary_df = st.session_state.get("col_summary_df")
            if col_summary_df is not None:
                try:
                    if col_summary_df["unique"].max() > max(20, 0.1 * df.shape[0]):
                        checklist.append("🔸 High-cardinality categories detected — consider target encoding or hashing.")
                except Exception:
             # If something goes wrong reading the column summary, ignore and continue
                    pass
            # Always useful recommendation
            checklist.append("🔸 Scale numeric features if using distance-based models (SVM/KNN).")       
            # Display each item
            for it in checklist:
                st.markdown(f"- {it}")

             # ---------------- Compact Human-Friendly Summary ---------------- #
            st.markdown("---")
            st.markdown("## ✨ Quick Highlights (shareable)")

            try:
                rows, cols = df.shape
                mem = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
                total_cells = max(1, rows * cols)
                missing_cells = int(df.isnull().sum().sum())
                missing_pct = round(missing_cells / total_cells * 100, 2)

                top_missing = (df.isnull().mean() * 100).sort_values(ascending=False).head(5)
                top_missing_list = [f"{c} ({round(p,2)}%)" for c, p in top_missing.items() if p > 0]
                top_missing_str = ", ".join(top_missing_list) if top_missing_list else "None"

                numeric_count = len(df.select_dtypes(include="number").columns)
                cat_count = len(df.select_dtypes(exclude="number").columns)

                # skewed
                skewed = []
                for c in df.select_dtypes(include="number").columns:
                    try:
                        s = df[c].dropna()
                        if s.shape[0] > 2 and abs(s.skew()) > 1:
                            skewed.append(c)
                    except:
                        pass
                skewed_str = ", ".join(skewed[:6]) if skewed else "None"

                # correlations
                strong_corrs = []
                num = df.select_dtypes(include="number")
                if num.shape[1] >= 2:
                    corr = num.corr()
                    for a in corr.columns:
                        for b in corr.columns:
                            if a < b:
                                v = corr.loc[a, b]
                                if abs(v) >= 0.6:
                                    strong_corrs.append((a, b, round(v, 2)))
                strong_corrs = sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True)[:6]

                if strong_corrs:
                    corr_text = "\n".join([f"• {a} ↔ {b} (corr = {v})" for a, b, v in strong_corrs])
                else:
                    corr_text = "None"

                high_card = [c for c in df.select_dtypes(exclude="number").columns if df[c].nunique() > max(20, 0.1 * rows)]
                high_card_str = ", ".join(high_card[:6]) if high_card else "None"

                recs = []
                if missing_pct > 3:
                    recs.append("Handle missing values (impute or drop).")
                if df.duplicated().sum() > 0:
                    recs.append("Remove duplicate rows.")
                if skewed:
                    recs.append("Normalize/transform highly skewed numeric columns.")
                if high_card:
                    recs.append("Encode high-cardinality categorical columns carefully.")
                recs.append("Dataset is ready for visualization & preprocessing.")

                # prepare recommendation lines *before* f-string
                rec_lines = "- " + "\n- ".join(recs)

                compact_md = f"""
**📌 Dataset:** {rows:,} rows × {cols} columns  
**💾 Memory usage:** {mem} MB  
**⚠️ Missing values:** {missing_pct}% of all cells  
**🔍 Columns with highest missing:** {top_missing_str}  
**🔢 Numeric columns:** {numeric_count}  
**🔠 Categorical columns:** {cat_count}  
**📈 Skewed numeric columns:** {skewed_str}  
**🔗 Strong correlations:**  
{corr_text}  
**🏷️ High-cardinality categorical columns:** {high_card_str}  

**🧹 Recommended actions:**  
{rec_lines}
"""

                # <-- IMPORTANT: actually display the compact summary
                st.markdown(compact_md)

                # optional: save to session so it persists across reruns
                st.session_state["quick_highlights"] = compact_md

            except Exception as e:
                st.error(f"Failed to build quick highlights: {e}")


# ---------------- About ---------------- #
elif selected_tab == "ℹ️ About":
    with st.container():
        st.markdown("<h2 style='text-align:;'>ℹ️ DataPilot AI Studio</h2>", unsafe_allow_html=True)

        # Main App Logo
        colA, colB, colC = st.columns([1,2,1])
        with colB:
            st.image("assets/Matrix logo.png", width=180, caption="MatrixLab AI Studio")

        st.markdown("""
Welcome to **DataPilot AI Studio** — your all-in-one, intelligent, interactive CSV analytics platform designed for  
**students, data analysts, researchers, and working professionals.**  
This tool converts raw datasets into **clean, structured, and insightful summaries within seconds**, with zero coding required.

---

## 🚀 What This Application Can Do

### 🔹 1. **Upload Any Dataset (CSV or Excel)**
- Instantly preview your dataset  
- Auto-detect column types  
- Fix column naming issues  

---

### 🔹 2. **⚡ One-Click Smart Cleaning**
Apply essential cleaning tasks instantly:
- Remove duplicates  
- Auto-fill missing values  
- Fix column names  
- Convert data types  
- Handle outliers (Cap / Remove)  
- Normalize or standardize numeric columns  

Perfect for **beginners** who want quick cleaning and **professionals** needing a fast preprocessing layer.

---

### 🔹 3. **🧹 Advanced Data Processing**
A more controlled, customizable cleaning engine:
- Per-column filling  
- Datetime conversion  
- Outlier removal using IQR  
- Column summary  
- Export cleaned dataset  
- Auto quality scoring  

---

### 🔹 4. **📊 Visualization (Basic & Advanced)**
Interactive plots powered by Matplotlib/Seaborn:
- Histograms, Boxplots, KDE  
- Scatterplots  
- Correlation heatmap  
- Bar & Pie charts  
- Pair plots  
- Time-series viewer  
- Category frequency explorer  
- Outlier explorer  

Designed for learning, presentations & quick insights.

---

### 🔹 5. **🔎 Explain My Dataset with AI**
Get an **attractive, emoji-rich summary** instantly:
- Executive snapshot  
- Missing value analysis  
- Skewness & distribution check  
- Correlation findings  
- High-cardinality detection  
- Actionable recommendations  
- Shareable Quick Highlights section  

No external API required — everything is generated locally.

---

### 🔹 6. **📦 Pipeline Management**
Save or load:
- Encoders  
- Scalers  
- Column schemas  
- Preprocessing workflows  

---

### 🔹 7. **🧠 SQL Query Builder**
Run SQL queries directly on your dataset:
A great learning tool for students + powerful filtering for professionals.

---

## 🎯 Why This App Is Special?
- ⭐ **Zero coding needed**  
- ⭐ **Beautiful UI with animations and icons**  
- ⭐ **Beginner-friendly, expert-powerful**  
- ⭐ **Complete EDA + cleaning in one place**  
- ⭐ **Fast, local, lightweight — no paid API required**  
- ⭐ **Perfect for college projects, research & analytics teams**

---

## 🏗️ Tech Behind the App
- **Python, Streamlit**  
- **Pandas, NumPy**  
- **Matplotlib / Seaborn**  
- **Joblib**  
- **Custom Data Cleaning Engine**  
- **Custom Attractive Summary Generator**  

---

## ❤️ Special Note
This application is designed to **simplify data analysis**,  
**accelerate learning**, and **empower professionals** to make data-driven decisions effortlessly.

Feel free to explore, experiment, and innovate.  
MatrixLab AI Studio is built for **your creativity**.
        """)

        # ---------------- Developed By Section ---------------- #
        st.markdown("---")
        st.markdown("## 👨‍💻 Developed By")

        # Center Team Logo
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.image("assets/Creative.png", width=180)
        
        

        st.markdown("""
### <div style='text-align:;'>**CREATIVE ENGE~NEARS**</div>
<div style='text-align:; font-size:18px;'>
 
        
👨‍💻 **Prasen Nimje**  
🔗 <a href='https://github.com/Prasen8' target='_blank'>GitHub Profile</a>

👨‍💻 **Mahesh Khumkar**  
🔗 <a href='https://github.com/MShriK17' target='_blank'>GitHub Profile</a>
</div>

<br>

Passionately building intelligent tools for the next generation of AI developers and data scientists.
</p>
""", unsafe_allow_html=True)

st.markdown("---")



# ---------------- Footer ---------------- #
st.markdown("---")
st.markdown("##### Your end-to-end AI & ML studio for modern data science", unsafe_allow_html=True)
st.caption("🔵 MatrixLab AI Studio v2.0 | CREATIVE ENGE~NEARS")
