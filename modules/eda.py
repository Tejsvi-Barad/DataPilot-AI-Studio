import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class EDAAnalyzer:
    def read_file(self, uploaded_file):
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif filename.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            raise ValueError("Unsupported file format.")

    def get_data_summary(self, df):
        summary = df.describe()
        nulls = df.isnull().sum().to_frame("Missing Values")
        dtypes = df.dtypes.to_frame("Data Type")
        return summary, nulls, dtypes

    def plot_histograms(self, df, top_n=20):
        num_cols = df.select_dtypes(include='number').columns[:top_n]
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f'Histogram of {col}')
            st.pyplot(fig)

    def plot_boxplots(self, df, top_n=20):
        num_cols = df.select_dtypes(include='number').columns[:top_n]
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f'Boxplot of {col}')
            st.pyplot(fig)

    def plot_violinplots(self, df, top_n=20):
        num_cols = df.select_dtypes(include='number').columns[:top_n]
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.violinplot(y=df[col], ax=ax)
            ax.set_title(f'Violin Plot of {col}')
            st.pyplot(fig)

    def plot_density(self, df, top_n=20):
        num_cols = df.select_dtypes(include='number').columns[:top_n]
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.kdeplot(df[col].dropna(), fill=True, ax=ax)
            ax.set_title(f'Density Plot of {col}')
            st.pyplot(fig)

    def plot_heatmap(self, df):
        num_df = df.select_dtypes(include='number')
        if num_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical columns to compute correlation.")

    def plot_scatterplots(self, df, x_col=None, y_col=None):
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)
        else:
            st.warning("⚠️ Invalid column selection.")

    def plot_pairplot(self, df):
        num_df = df.select_dtypes(include='number')
        if num_df.shape[1] > 1:
            fig = sns.pairplot(num_df)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for pairplot.")

    def plot_barplots(self, df, top_n=15):
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            top_values = df[col].value_counts().nlargest(top_n)
            fig, ax = plt.subplots()
            top_values.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'Bar Plot of {col} (Top {top_n})')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            st.pyplot(fig)

    def plot_piecharts(self, df, top_n=10):
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            value_counts = df[col].value_counts().nlargest(top_n)
            fig, ax = plt.subplots()
            value_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
            ax.set_title(f'Pie Chart of {col} (Top {top_n})')
            st.pyplot(fig)
