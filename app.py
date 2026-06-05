import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Hierarchical Clustering",
    layout="wide"
)

st.title("Hierarchical Clustering Dashboard")

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Load Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    # EDA Section
    # -------------------------------
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().reset_index().rename(
        columns={"index": "Column", 0: "Missing Values"}
    ))

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # -------------------------------
    # Data Preprocessing
    # -------------------------------
    processed_df = df.copy()

    for col in processed_df.columns:
        if processed_df[col].dtype == "object":
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(
                processed_df[col].astype(str)
            )

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(processed_df)

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=processed_df.columns
    )

    # -------------------------------
    # Correlation Heatmap
    # -------------------------------
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        scaled_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------------
    # Distribution Plot
    # -------------------------------
    st.subheader("Feature Distribution")

    selected_column = st.selectbox(
        "Select Column",
        scaled_df.columns
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        scaled_df[selected_column],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # -------------------------------
    # Dendrogram
    # -------------------------------
    st.header("Hierarchical Clustering")

    st.subheader("Dendrogram")

    linkage_method = st.selectbox(
        "Linkage Method",
        ["ward", "complete", "average", "single"]
    )

    linkage_matrix = linkage(
        scaled_df,
        method=linkage_method
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    dendrogram(
        linkage_matrix,
        ax=ax,
        truncate_mode="level",
        p=5
    )

    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    st.pyplot(fig)

    # -------------------------------
    # Clustering
    # -------------------------------
    from sklearn.cluster import AgglomerativeClustering

    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    clusters = model.fit_predict(scaled_df)

    result_df = df.copy()
    result_df["Cluster"] = clusters

    st.subheader("Clustered Dataset")
    st.dataframe(result_df)

    # -------------------------------
    # Cluster Distribution
    # -------------------------------
    st.subheader("Cluster Distribution")

    cluster_counts = (
        result_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(cluster_counts)

    # -------------------------------
    # Download Results
    # -------------------------------
    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clustered CSV",
        data=csv,
        file_name="hierarchical_clustering_output.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file.")