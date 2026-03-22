import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from ydata_profiling import ProfileReport

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from pycaret.classification import *
from pycaret.regression import *

st.set_page_config(page_title="Streamlit AutoML Platform", layout="wide")
st.title("🚀 Streamlit-Driven AutoML System")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset Loaded Successfully!")

    tab1, tab2, tab3 = st.tabs(["📊 AutoEDA", "🤖 AutoML", "📌 Clustering"])

    # ===============================
    # AUTO EDA
    # ===============================
    with tab1:
        st.header("Automated Data Profiling")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
            st.pyplot(fig)

        st.subheader("Full Profiling Report")
        profile = ProfileReport(df, explorative=True)
        st.components.v1.html(profile.to_html(), height=800, scrolling=True)

    # ===============================
    # AUTOML
    # ===============================
    with tab2:
        st.header("Automated Machine Learning")

        target = st.selectbox("Select Target Column", df.columns)

        problem_type = "Classification" if df[target].dtype == "object" or df[target].nunique() < 20 else "Regression"
        st.info(f"Detected Problem Type: {problem_type}")

        if st.button("Run AutoML Pipeline"):

            if problem_type == "Classification":

                setup(data=df,
                      target=target,
                      session_id=42,
                      fold=5,
                      verbose=False)

                st.info("Comparing Models...")
                best_model = compare_models()

                st.success("Best Model Selected")
                st.write(best_model)

                st.info("Tuning Best Model...")
                tuned_model = tune_model(best_model)

                final_model = finalize_model(tuned_model)

                save_model(final_model, "best_classification_model")

                st.success("Final Classification Model Saved!")

                plot_model(best_model, plot='confusion_matrix')

            else:

                setup(data=df,
                      target=target,
                      session_id=42,
                      fold=5,
                      
                      verbose=False)

                st.info("Comparing Models...")
                best_model = compare_models()

                st.success("Best Model Selected")
                st.write(best_model)

                st.info("Tuning Best Model...")
                tuned_model = tune_model(best_model)

                final_model = finalize_model(tuned_model)

                save_model(final_model, "best_regression_model")

                st.success("Final Regression Model Saved!")

                plot_model(best_model, plot='feature')

    # ===============================
    # CLUSTERING
    # ===============================
    with tab3:
        st.header("K-Means Clustering")

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.empty:
            st.warning("No numeric columns available.")
        else:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            k = st.slider("Select Number of Clusters (k)", 2, 10, 3)

            if st.button("Run Clustering"):

                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(scaled_data)

                df["Cluster"] = labels
                st.dataframe(df)

                sil = silhouette_score(scaled_data, labels)
                st.write(f"Silhouette Score: {sil:.3f}")

                fig = px.scatter(df,
                                 x=numeric_df.columns[0],
                                 y=numeric_df.columns[1],
                                 color="Cluster",
                                 title="Cluster Visualization")

                st.plotly_chart(fig)

else:
    st.info("Upload a dataset to begin.")