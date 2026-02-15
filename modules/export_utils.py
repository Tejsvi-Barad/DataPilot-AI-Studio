import pandas as pd
import joblib
import os

class ExportManager:
    def __init__(self, export_dir="exports"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_csv(self, df, filename="cleaned_data.csv"):
        path = os.path.join(self.export_dir, filename)
        df.to_csv(path, index=False)
        return path

    def export_model(self, model, filename="trained_model.pkl"):
        path = os.path.join(self.export_dir, filename)
        joblib.dump(model, path)
        return path

    def export_plot(self, fig, filename="plot.png"):
        path = os.path.join(self.export_dir, filename)
        fig.savefig(path)
        return path


# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from modules.eda import EDAAnalyzer
# from modules.data_cleaner import DataCleaner
# from modules.model_train import MLTrainer

# import requests
# import time
# import json
# import streamlit.components.v1 as components  # 👈 for custom Lottie renderer


# # ---------------- LOTTIE HELPER (no streamlit_lottie dependency) ---------------- #

# def st_lottie(lottie_source, height=400, key=None):
#     """
#     Minimal Lottie renderer using the lottie-player web component.

#     lottie_source: can be either
#         - a URL string to a .json Lottie file
#         - or a Python dict with Lottie JSON
#     """
#     # If source is a URL string, embed directly
#     if isinstance(lottie_source, str):
#         url = lottie_source
#         html = f"""
#         <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
#         <lottie-player
#             src="{url}"
#             background="transparent"
#             speed="1"
#             style="width: 100%; height: {height}px;"
#             loop
#             autoplay
#         ></lottie-player>
#         """
#         components.html(html, height=height + 20)
#         return

#     # If source is JSON/dict, embed as animationData
#     try:
#         animation_data = json.dumps(lottie_source)
#         html = f"""
#         <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
#         <div id="lottie-{key or 'anim'}"></div>
#         <script>
#           const container = document.getElementById("lottie-{key or 'anim'}");
#           const player = document.createElement("lottie-player");
#           player.background = "transparent";
#           player.speed = 1;
#           player.style.width = "100%";
#           player.style.height = "{height}px";
#           player.loop = true;
#           player.autoplay = true;
#           container.appendChild(player);
#           const animationData = {animation_data};
#           player.load(animationData);
#         </script>
#         """
#         components.html(html, height=height + 20)
#     except Exception:
#         st.warning("⚠️ Could not render Lottie animation.")


# def load_lottie_url(url: str):
#     """(Optional) Load Lottie JSON from URL if you want to use JSON instead of src URL."""
#     try:
#         r = requests.get(url, timeout=8)
#         if r.status_code != 200:
#             return None
#         return r.json()
#     except Exception:
#         return None


# # ------------------------ SESSION + WELCOME SCREEN ------------------------ #

# # Set session state for first-time entry
# if "entered_app" not in st.session_state:
#     st.session_state.entered_app = False

# # Show welcome screen if not entered
# if not st.session_state.entered_app:
#     st.set_page_config(page_title="MatrixLab AI Studio",
#                        layout="wide",
#                        initial_sidebar_state="collapsed")

#     st.markdown(
#         """
#         <style>
#         .block-container {
#             padding-top: 0rem;
#             padding-bottom: 0rem;
#         }
#         .top {
#             display: flex;
#             flex-direction: column;
#             align-items: top;
#             justify-content: top;
#             height: 100vh;
#             text-align: top;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     with st.container():
#         st.markdown('<div class="center">', unsafe_allow_html=True)

#         # You can either pass the URL directly:
#         st_lottie(
#             "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json",
#             height=350,
#             key="welcome_animation"
#         )

#         # Or if you want to load JSON explicitly:
#         # lottie_json = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")
#         # st_lottie(lottie_json, height=350, key="welcome_animation")

#         st.markdown("<h1 style='color:white;'>👋 Welcome to MatrixLab AI Studio</h1>", unsafe_allow_html=True)

#         if st.button("🚀 Tap to Enter"):
#             st.session_state.entered_app = True
#             st.rerun()
#         st.markdown('</div>', unsafe_allow_html=True)

#     st.stop()

# # ------------------------ MAIN APP LAYOUT ------------------------ #

# # Configure page (Streamlit warns if set_page_config called twice, but it still works)
# st.set_page_config(
#     page_title="Matrix AI CSV Analyser",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("<h1 style='text-align:center;'>🧠 MatrixLab AI Studio</h1>", unsafe_allow_html=True)
# st.markdown("##### Your end-to-end AI & ML studio for modern data science", unsafe_allow_html=True)

# st.sidebar.title("📍 Navigation")
# selected_tab = st.sidebar.radio("Choose your workflow step:", [
#     "📤 Data Upload",
#     "⚡ One-Click Smart Cleaning",
#     "🧹 Data Processing",
#     "📊 Visualization",
#     "📊 Visualization (Advanced)",
#     "🔎 Explain My Dataset (AI)",
#     "📦 Pipeline Management",
#     "ℹ️ About"
# ])

# if st.sidebar.button("👈 Back to Welcome"):
#     st.session_state.entered_app = False
#     st.rerun()

# st.markdown("---")
# status_cols = st.columns(4)
# status_cols[0].markdown("📥 **Data Loaded:** " + ("✅" if st.session_state.data is not None else "❌"))
# status_cols[1].markdown("🧹 **Data Processed:** " + ("✅" if st.session_state.cleaned_data is not None else "⚠️"))
# status_cols[2].markdown("💾 **Pipeline Saved:** " + ("✅" if st.session_state.pipeline_obj is not None else "❌"))
# status_cols[3].markdown("🔎 **Insights Ready:** " + ("✅" if st.session_state.cleaned_data is not None or st.session_state.data is not None else "⚠️"))
# st.markdown("---")


# # ---------------- Utility helpers ---------------- #
# def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
#     # convert to snake_case, remove strange chars
#     new_cols = []
#     for c in df.columns:
#         s = str(c).strip()
#         s = s.replace(" ", "_").replace("-", "_")
#         s = "".join(ch for ch in s if ch.isalnum() or ch == "_")
#         s = s.lower()
#         new_cols.append(s)
#     df.columns = new_cols
#     return df

# def auto_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
#     for col in df.columns:
#         if df[col].isnull().any():
#             if pd.api.types.is_numeric_dtype(df[col]):
#                 df[col] = df[col].fillna(df[col].mean())
#             else:
#                 mode = df[col].mode()
#                 df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "missing")
#     return df

# def convert_datatypes_auto(df: pd.DataFrame) -> pd.DataFrame:
#     # try numeric
#     for col in df.columns:
#         if df[col].dtype == object:
#             # try int/float
#             try:
#                 df[col] = pd.to_numeric(df[col], errors="ignore")
#             except Exception:
#                 pass
#     # try detect dates
#     for col in df.columns:
#         if df[col].dtype == object:
#             try:
#                 parsed = pd.to_datetime(df[col], errors="coerce")
#                 # if many values parsed, accept conversion
#                 if parsed.notna().sum() / max(1, len(parsed)) > 0.6:
#                     df[col] = parsed
#             except Exception:
#                 pass
#     return df

# def drop_useless_columns(df: pd.DataFrame, missing_thresh=0.95) -> pd.DataFrame:
#     # drop columns with > missing_thresh fraction missing and zero variance columns
#     drop_cols = []
#     for col in df.columns:
#         if df[col].isnull().mean() > missing_thresh:
#             drop_cols.append(col)
#         elif df[col].nunique() <= 1:
#             drop_cols.append(col)
#     if drop_cols:
#         df = df.drop(columns=drop_cols)
#     return df

# def auto_handle_outliers(df: pd.DataFrame, action="remove", iqr_multiplier=1.5):
#     # action: "remove" or "cap"
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     mask = pd.Series(True, index=df.index)
#     for col in numeric_cols:
#         q1 = df[col].quantile(0.25)
#         q3 = df[col].quantile(0.75)
#         iqr = q3 - q1
#         lower = q1 - iqr_multiplier * iqr
#         upper = q3 + iqr_multiplier * iqr
#         if action == "remove":
#             mask &= (df[col] >= lower) & (df[col] <= upper)
#         else:
#             df[col] = np.where(df[col] < lower, lower, df[col])
#             df[col] = np.where(df[col] > upper, upper, df[col])
#     if action == "remove":
#         df = df.loc[mask]
#     return df

# def scale_numeric(df: pd.DataFrame, method="standard"):
#     numeric_cols = df.select_dtypes(include="number").columns.tolist()
#     if not numeric_cols:
#         return df, None
#     if method == "standard":
#         from sklearn.preprocessing import StandardScaler
#         scaler = StandardScaler()
#         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#         return df, scaler
#     elif method == "minmax":
#         from sklearn.preprocessing import MinMaxScaler
#         scaler = MinMaxScaler()
#         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
#         return df, scaler
#     else:
#         return df, None

# def compute_quality(df: pd.DataFrame):
#     rows = max(1, df.shape[0]); cols = max(1, df.shape[1])
#     total = rows * cols
#     missing_cells = int(df.isnull().sum().sum())
#     missing_pct = (missing_cells / total) * 100.0
#     duplicates = int(df.duplicated().sum()); dup_ratio = duplicates / rows
#     zero_var = int((df.nunique() <= 1).sum()); zratio = zero_var / cols
#     cat_cols = df.select_dtypes(exclude="number").columns.tolist()
#     high_card = sum(1 for c in cat_cols if rows and df[c].nunique() / rows > 0.9)
#     high_card_ratio = high_card / (len(cat_cols) if len(cat_cols) else 1)
#     score = 100.0
#     score -= missing_pct * 0.6
#     score -= dup_ratio * 40.0
#     score -= zratio * 25.0
#     score -= high_card_ratio * 15.0
#     if rows < 10: score -= 10.0
#     elif rows < 50: score -= 5.0
#     score = max(0.0, min(100.0, round(score,2)))
#     label = "Excellent ✅" if score>=85 else ("Good 👍" if score>=70 else ("Fair ⚠️" if score>=50 else "Poor ❌"))
#     return dict(score=score, label=label, missing_pct=round(missing_pct,2), duplicates=duplicates, zero_var=zero_var, high_card=high_card, mem_mb=round(df.memory_usage(deep=True).sum()/1024**2,2))

# # ------------------------ TABS LOGIC ------------------------ #
# # ---------------- Data Upload ---------------- #
# if selected_tab == "📤 Data Upload":
#     st.subheader("📤 Upload Dataset")
#     uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
#     if uploaded_file:
#         eda = EDAAnalyzer()
#         df = eda.read_file(uploaded_file)
#         st.session_state.data = df
#         st.session_state.cleaned_data = None  # reset previous cleaned
#         st.success("✅ File loaded successfully!")
#         st.dataframe(df.head())

# # ---------------- One-Click Smart Cleaning ---------------- #
# elif selected_tab == "⚡ One-Click Smart Cleaning":
#     st.subheader("⚡ One-Click Smart Cleaning / Quick fixes")
#     if st.session_state.data is None:
#         st.warning("Please upload a dataset first.")
#     else:
#         st.markdown("Use these quick actions to auto-clean your dataset. Each button performs a safe, transparent operation.")
#         df0 = st.session_state.data.copy()

#         col1, col2, col3 = st.columns(3)
#         if col1.button("Auto Clean Dataset (recommended)"):
#             try:
#                 df = df0.copy()
#                 df = fix_column_names(df)
#                 df = auto_fill_missing(df)
#                 df = convert_datatypes_auto(df)
#                 df = drop_useless_columns(df)
#                 df = auto_handle_outliers(df, action="cap")
#                 df, scaler_obj = scale_numeric(df, method="standard")
#                 st.session_state.cleaned_data = df
#                 st.success("Auto clean finished and saved to session.cleaned_data")
#                 st.dataframe(df.head())
#             except Exception as e:
#                 st.error(f"Auto clean failed: {e}")

#         if col1.button("Remove duplicates"):
#             df = df0.drop_duplicates()
#             st.session_state.cleaned_data = df
#             st.success("Duplicates removed.")
#             st.dataframe(df.head())

#         if col1.button("Fix column names"):
#             df = fix_column_names(df0)
#             st.session_state.data = df  # update original too
#             st.success("Column names normalized (snake_case).")
#             st.dataframe(df.head())

#         if col2.button("Auto-fill missing values"):
#             df = auto_fill_missing(df0.copy())
#             st.session_state.cleaned_data = df
#             st.success("Missing values auto-filled (mean/mode).")
#             st.dataframe(df.head())

#         if col2.button("Convert datatypes (auto)"):
#             df = convert_datatypes_auto(df0.copy())
#             st.session_state.cleaned_data = df
#             st.success("Datatypes converted where possible (numbers/dates).")
#             st.dataframe(df.head())

#         if col2.button("Drop useless columns"):
#             df = drop_useless_columns(df0.copy())
#             st.session_state.cleaned_data = df
#             st.success("Dropped columns with >95% missing or zero variance.")
#             st.dataframe(df.head())

#         if col3.button("Auto-handle outliers (cap)"):
#             df = auto_handle_outliers(df0.copy(), action="cap")
#             st.session_state.cleaned_data = df
#             st.success("Outliers capped (IQR).")
#             st.dataframe(df.head())

#         if col3.button("Auto-handle outliers (remove)"):
#             df = auto_handle_outliers(df0.copy(), action="remove")
#             st.session_state.cleaned_data = df
#             st.success("Outlier rows removed (IQR).")
#             st.dataframe(df.head())

#         if col3.button("Standardize numeric columns"):
#             df, scaler_obj = scale_numeric(df0.copy(), method="standard")
#             st.session_state.cleaned_data = df
#             st.session_state.pipeline_obj = st.session_state.get("pipeline_obj", {})
#             st.success("Numeric columns standardized (zero mean, unit variance).")
#             st.dataframe(df.head())

#         if st.button("Normalize numeric columns (MinMax)"):
#             df, scaler_obj = scale_numeric(df0.copy(), method="minmax")
#             st.session_state.cleaned_data = df
#             st.success("Numeric columns normalized to [0,1].")
#             st.dataframe(df.head())

#         st.markdown("---")
#         st.info("Tip: after using One-Click actions, open **Data Processing** to apply more controlled cleaning and save pipeline.")


# # ---------------- Data Processing ---------------- #
# elif selected_tab == "🧹 Data Processing":
#     st.subheader("🧹 Auto Data Cleaning (advanced)")
#     if st.session_state.cleaned_data is None and st.session_state.data is None:
#         st.warning("Please upload a dataset (Data Upload) or use One-Click Smart Cleaning first.")
#     else:
#         # prefer cleaned_data if present
#         df_base = st.session_state.cleaned_data.copy() if st.session_state.cleaned_data is not None else st.session_state.data.copy()
#         cleaner = DataCleaner()
#         drop_duplicates = st.checkbox("🗑️ Remove Duplicates", True)
#         fill_missing = st.checkbox("💧 Fill Missing Values", True)
#         convert_types = st.checkbox("🕒 Convert Text to Date/Time", True)

#         if st.button("🚀 Clean Data"):
#             try:
#                 cleaned_df, orig_shape, new_shape = cleaner.clean(df_base.copy(), drop_duplicates, fill_missing, convert_types)
#                 st.session_state.cleaned_data = cleaned_df
#                 st.success("✅ Data cleaned successfully!")
#                 st.write(f"🔹 Original shape: {orig_shape}")
#                 st.write(f"🔹 New shape: {new_shape}")
#                 st.dataframe(cleaned_df.head())

#                 csv = cleaned_df.to_csv(index=False, encoding='utf-8', errors='replace')
#                 st.download_button("⬇️ Download Cleaned Data", data=csv, file_name="cleaned_data.csv", mime="text/csv")

#                 # Quality scoring (simple)
#                 quality = compute_quality(cleaned_df)
#                 st.markdown("## 📊 Exploratory Data Analysis — Auto Summary")
#                 col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
#                 col1.metric("📏 Rows", f"{cleaned_df.shape[0]:,}")
#                 col2.metric("📐 Columns", cleaned_df.shape[1])
#                 col3.metric("🧠 Memory (MB)", f"{quality['mem_mb']:.2f}")
#                 col4.metric("🧼 Missing (%)", f"{quality['missing_pct']:.2f}%")
#                 col5.metric("♻️ Duplicates", quality['duplicates'])
#                 col6.metric("🧱 Zero Var Cols", quality['zero_var'])
#                 col7.metric("📊 Quality Score", f"{quality['score']}/100 — {quality['label']}")

#                 numerics = cleaned_df.select_dtypes(include="number").columns.tolist()
#                 categoricals = cleaned_df.select_dtypes(exclude="number").columns.tolist()
#                 st.markdown(f"🔢 **Numeric Columns ({len(numerics)}):** `{', '.join(numerics[:8]) + ('...' if len(numerics) > 8 else '')}`")
#                 st.markdown(f"🔠 **Categorical Columns ({len(categoricals)}):** `{', '.join(categoricals[:8]) + ('...' if len(categoricals) > 8 else '')}`")

#                 with st.expander("📈 Show Descriptive Statistics"):
#                     st.dataframe(cleaned_df.describe(include="all").T)
#             except Exception as e:
#                 st.error(f"❌ Data processing failed: {e}")

# # ---------------- Visualization  ---------------- #
# elif selected_tab == "📊 Visualization":
#     st.subheader("📊 Explore Visualizations")
#     if st.session_state.cleaned_data is None:
#         st.warning("🧹 Please clean your data first.")
#     else:
#         df = st.session_state.cleaned_data
#         eda = EDAAnalyzer()

#         chart_type = st.selectbox("🎛️ Select a chart type:", [
#             "Histogram", "Box Plot", "Violin Plot", "Density Plot",
#             "Scatter Plot", "Correlation Heatmap", "Pair Plot",
#             "Bar Plot", "Pie Chart"
#         ])

#         if chart_type == "Scatter Plot":
#             x_axis = st.selectbox("📊 Select X-axis", df.columns.tolist(), key="scatter_x_axis")
#             y_axis = st.selectbox("📈 Select Y-axis", df.columns.tolist(), key="scatter_y_axis")

#             if x_axis and y_axis:
#                 if st.button("📈 Generate Scatter Plot"):
#                     try:
#                         eda.plot_scatterplots(df, x_col=x_axis, y_col=y_axis)
#                     except Exception as e:
#                         st.error(f"❌ Error generating scatter plot: {e}")
#             else:
#                 st.info("ℹ️ Please select both X and Y axes.")

#         else:
#             if st.button(f"📈 Generate {chart_type}"):
#                 try:
#                     chart_map = {
#                         "Histogram": eda.plot_histograms,
#                         "Box Plot": eda.plot_boxplots,
#                         "Violin Plot": eda.plot_violinplots,
#                         "Density Plot": eda.plot_density,
#                         "Correlation Heatmap": eda.plot_heatmap,
#                         "Pair Plot": eda.plot_pairplot,
#                         "Bar Plot": eda.plot_barplots,
#                         "Pie Chart": eda.plot_piecharts,
#                     }
#                     chart_map[chart_type](df)
#                 except Exception as e:
#                     st.error(f"❌ Error generating {chart_type}: {e}")

# # ---------------- Visualization (Advanced)  ---------------- #
# elif selected_tab == "📊 Visualization (Advanced)":
#     st.subheader("📊 Advanced Visualizations")
#     if st.session_state.cleaned_data is None:
#         st.warning("Please clean/process your data first (Data Processing).")
#     else:
#         df = st.session_state.cleaned_data.copy()
#         eda = EDAAnalyzer()

#         vis_mode = st.selectbox("Visualization Mode", [
#             "Pairwise Correlation Explorer",
#             "Feature Distribution Analyzer",
#             "Category Frequency Visualizer",
#             "Time-Series Trend Viewer",
#             "Boxplot Outlier Explorer"
#         ])

#         numeric_cols = df.select_dtypes(include="number").columns.tolist()
#         categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
#         datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

#         # Pairwise Correlation Explorer
#         if vis_mode == "Pairwise Correlation Explorer":
#             st.markdown("**Pairwise Correlation Explorer**")
#             cols = st.multiselect("Select numeric columns (min 2)", options=numeric_cols, default=numeric_cols[:6])
#             method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
#             threshold = st.slider("Show pairs with |corr| ≥", 0.0, 1.0, 0.6, step=0.05)
#             if len(cols) >= 2:
#                 sub = df[cols].dropna(how="all")
#                 corr = sub.corr(method=method)
#                 fig, ax = plt.subplots(figsize=(8,6))
#                 sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
#                 ax.set_title(f"{method.title()} Correlation Heatmap")
#                 st.pyplot(fig)

#                 pairs = []
#                 for i in range(len(corr.columns)):
#                     for j in range(i+1, len(corr.columns)):
#                         a,b = corr.columns[i], corr.columns[j]
#                         val = corr.iloc[i,j]
#                         if abs(val) >= threshold:
#                             pairs.append((a,b,round(val,3)))
#                 if pairs:
#                     st.table(pd.DataFrame(pairs, columns=["col1","col2",f"{method}_corr"]))
#                 else:
#                     st.info("No pairs above threshold.")
#             else:
#                 st.info("Pick at least 2 numeric columns.")

#         # Feature Distribution Analyzer
#         elif vis_mode == "Feature Distribution Analyzer":
#             st.markdown("**Feature Distribution Analyzer**")
#             cols = st.multiselect("Pick numeric columns", options=numeric_cols, default=numeric_cols[:3])
#             plot_kind = st.selectbox("Plot", ["Histogram (overlay)", "KDE (overlay)", "Boxplot (side-by-side)"])
#             bins = st.slider("Bins", 5, 200, 30)
#             if cols:
#                 fig, ax = plt.subplots(figsize=(8,5))
#                 if plot_kind == "Histogram (overlay)":
#                     for c in cols:
#                         sns.histplot(df[c].dropna(), bins=bins, stat="density", element="step", alpha=0.35, label=c, ax=ax)
#                     ax.legend(); ax.set_title("Overlay Histograms")
#                 elif plot_kind == "KDE (overlay)":
#                     for c in cols:
#                         sns.kdeplot(df[c].dropna(), label=c, fill=False, ax=ax)
#                     ax.legend(); ax.set_title("Overlay KDEs")
#                 else:
#                     sns.boxplot(data=df[cols], orient="v", ax=ax); ax.set_title("Boxplots")
#                 st.pyplot(fig)
#                 st.write(df[cols].describe().T)
#             else:
#                 st.info("Select one or more numeric columns.")

#         # Category Frequency Visualizer
#         elif vis_mode == "Category Frequency Visualizer":
#             st.markdown("**Category Frequency Visualizer**")
#             if not categorical_cols:
#                 st.info("No categorical columns found.")
#             else:
#                 cat = st.selectbox("Choose categorical column", options=categorical_cols)
#                 top_k = st.slider("Top k", 3, 50, 10)
#                 normalize = st.checkbox("Show relative frequency", value=False)
#                 vc = df[cat].fillna("<<MISSING>>").value_counts(normalize=normalize).head(top_k)
#                 fig, ax = plt.subplots(figsize=(8,4))
#                 sns.barplot(x=vc.values, y=vc.index, orient="h", ax=ax)
#                 ax.set_xlabel("Proportion" if normalize else "Count")
#                 ax.set_ylabel(cat)
#                 ax.set_title(f"Top {top_k} categories in {cat}")
#                 st.pyplot(fig)
#                 st.table(vc.to_frame("proportion" if normalize else "count"))

#         # Time-Series Trend Viewer
#         elif vis_mode == "Time-Series Trend Viewer":
#             st.markdown("**Time-Series Trend Viewer**")
#             pick = st.selectbox("Datetime column (or choose another column to parse)", options=(["--auto--"] + df.columns.tolist()))
#             if pick == "--auto--":
#                 autods = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
#                 if not autods:
#                     st.info("No datetime columns auto-detected. Convert a column to datetime in Data Processing.")
#                 else:
#                     pick = st.selectbox("Detected datetime columns", autods)
#             if pick and pick != "--auto--":
#                 try:
#                     tmp = df.copy()
#                     if not pd.api.types.is_datetime64_any_dtype(tmp[pick]):
#                         tmp[pick] = pd.to_datetime(tmp[pick], errors="coerce")
#                     values = tmp.select_dtypes(include="number").columns.tolist()
#                     value_cols = st.multiselect("Value columns", options=values, default=values[:1])
#                     resample_period = st.selectbox("Resample", ["None", "D", "W", "M"])
#                     if st.button("Plot Series"):
#                         ts = tmp.dropna(subset=[pick] + value_cols).sort_values(pick)
#                         if resample_period != "None" and value_cols:
#                             ts = ts.set_index(pick).resample(resample_period).mean().reset_index()
#                         fig, ax = plt.subplots(figsize=(10,4))
#                         for c in value_cols:
#                             ax.plot(ts[pick], ts[c], label=c)
#                         ax.legend(); ax.set_title("Time Series")
#                         st.pyplot(fig)
#                 except Exception as e:
#                     st.error(f"Time series error: {e}")

#         # Boxplot Outlier Explorer
#         elif vis_mode == "Boxplot Outlier Explorer":
#             st.markdown("**Boxplot Outlier Explorer**")
#             if not numeric_cols:
#                 st.info("No numeric columns available.")
#             else:
#                 col = st.selectbox("Choose numeric column", options=numeric_cols)
#                 multiplier = st.slider("IQR multiplier", 1.0, 4.0, 1.5)
#                 maxrows = st.number_input("Max sample outlier rows", min_value=5, max_value=500, value=20)
#                 if st.button("Show Outliers"):
#                     s = df[col].dropna()
#                     q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
#                     lower = q1 - multiplier * iqr; upper = q3 + multiplier * iqr
#                     mask = (df[col] < lower) | (df[col] > upper)
#                     n = int(mask.sum())
#                     fig, ax = plt.subplots(figsize=(6,3))
#                     sns.boxplot(x=s, ax=ax)
#                     ax.set_title(f"{col} — Outliers: {n}")
#                     st.pyplot(fig)
#                     st.write(f"Outliers flagged: {n}")
#                     if n>0:
#                         st.dataframe(df.loc[mask].head(maxrows))

# # ---------------- Explain My Dataset (AI) ---------------- #
# elif selected_tab == "🔎 Explain My Dataset (AI)":
#     st.subheader("🔎 Explain My Dataset (AI-powered)")

#     if st.session_state.cleaned_data is None and st.session_state.data is None:
#         st.warning("Please upload/clean a dataset first.")
#     else:
#         df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
#         st.markdown("### Local heuristic explanation (instant)")
#         if st.button("Generate local explanation"):
#             # build explanation
#             expl = []
#             expl.append(f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
#             expl.append(f"Memory usage: {round(df.memory_usage(deep=True).sum()/1024**2,2)} MB.")
#             miss_pct = round((df.isnull().sum().sum()/(max(1,df.shape[0]*df.shape[1])))*100,2)
#             expl.append(f"Overall missing values: {miss_pct}%. Top columns with missing: {', '.join((df.isnull().mean()*100).sort_values(ascending=False).head(5).index.tolist())}")
#             # numeric correlations
#             num = df.select_dtypes(include="number")
#             if num.shape[1] >= 2:
#                 corr = num.corr().abs()
#                 pairs=[]
#                 for i in range(len(corr.columns)):
#                     for j in range(i+1,len(corr.columns)):
#                         pairs.append((corr.columns[i], corr.columns[j], corr.iloc[i,j]))
#                 top = sorted(pairs, key=lambda x: x[2], reverse=True)[:5]
#                 expl.append("Top numeric correlations (abs): " + ", ".join([f"{a}-{b}:{round(s,2)}" for a,b,s in top if s>0.25]))
#             # categories
#             cat = df.select_dtypes(exclude="number")
#             if len(cat.columns)>0:
#                 high_card = [c for c in cat.columns if df[c].nunique()/max(1,df.shape[0]) > 0.9]
#                 if high_card:
#                     expl.append("High-cardinality categorical columns: " + ", ".join(high_card))
#             # recommendations
#             recs = []
#             if miss_pct > 5: recs.append("Impute or drop columns with >5% missing.")
#             if df.duplicated().sum() > 0: recs.append("Deduplicate rows if duplicates are artifacts.")
#             if any(num.skew().abs() > 1): recs.append("Consider transforming skewed numeric columns (log / box-cox).")
#             if not recs: recs.append("Dataset looks reasonably clean.")
#             expl.append("Recommendations: " + " ".join(recs))

#             text = "\n\n".join(expl)
#             st.session_state.insights_cache = text
#             st.success("Local explanation generated.")
#         if st.session_state.get("insights_cache"):
#             st.text_area("Explanation (editable)", value=st.session_state.insights_cache, height=220)

#         st.markdown("### Optional: Polished explanation via OpenAI (requires API key)")
#         api_key = st.text_input("OpenAI API Key (paste here to enable polishing). Keep blank to skip.", type="password")
#         prompt_template = st.text_area("Prompt to send to LLM (optional override)", value="", height=80)

#         if api_key and st.button("Polish explanation with OpenAI"):
#             if not OPENAI_AVAILABLE:
#                 st.error("OpenAI package not installed. Run `pip install openai` in your venv.")
#             else:
#                 try:
#                     openai.api_key = api_key
#                     base_text = st.session_state.get("insights_cache", "")
#                     if not base_text:
#                         st.info("Generate the local explanation first (button above).")
#                     else:
#                         prompt = prompt_template.strip() or f"Polish and elaborate the following dataset summary into a concise, clear report for a data analysis project:\n\n{base_text}"
#                         with st.spinner("Calling OpenAI..."):
#                             resp = openai.ChatCompletion.create(
#                                 model="gpt-4o-mini",  # change as desired
#                                 messages=[{"role":"user","content":prompt}],
#                                 max_tokens=800,
#                                 temperature=0.2
#                             )
#                             polished = resp.choices[0].message.content.strip()
#                             st.text_area("Polished explanation (from OpenAI)", value=polished, height=320)
#                 except Exception as e:
#                     st.error(f"OpenAI call failed: {e}")

# # ---------------- Pipeline Management ---------------- #
# elif selected_tab == "📦 Pipeline Management":
#     st.subheader("📦 Export / Import Preprocessing Pipeline")
#     st.write("Save or load preprocessing pipeline (encoders, scalers, features).")
#     if st.session_state.pipeline_obj is not None:
#         st.write("Pipeline currently in session.")
#         try:
#             b = joblib.dumps(st.session_state.pipeline_obj)
#             st.download_button("📥 Download current pipeline (joblib)", data=b, file_name="pipeline_session.joblib")
#         except Exception:
#             pass
#     up = st.file_uploader("Upload pipeline (.joblib/.pkl)", type=["joblib","pkl"])
#     if up:
#         try:
#             loaded = joblib.load(up)
#             st.session_state.pipeline_obj = loaded
#             st.success("Pipeline loaded into session.")
#         except Exception as e:
#             st.error(f"Failed to load pipeline: {e}")

# # ---------------- About ---------------- #
# elif selected_tab == "ℹ️ About":
#     st.subheader("ℹ️ About MatrixLab AI Studio — Insights edition")
#     st.write("""
#     - This edition focuses on EDA, one-click cleaning, advanced visualizations and AI-powered explanations.
#     - Model training/evaluation/prediction intentionally removed.
#     - Optional OpenAI polishing available if you provide an API key (must install `openai` package).
#     """)
#     st.markdown("- Tip: Use One-Click Smart Cleaning first for quick fixes, then Data Processing for controlled cleaning steps.")
#     st.markdown("- Contact: add contact info or links here.")

# # ---------------- Footer ---------------- #
# st.markdown("---")
# st.markdown("##### Your end-to-end AI & ML studio for modern data science", unsafe_allow_html=True)
# st.caption("🔵 MatrixLab AI Studio v2.0 | CREATIVE ENGE~NEARS")
