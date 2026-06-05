import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Hierarchical Clustering",
    layout="wide"
)

st.title("Hierarchical Clustering Dashboard")

# -----------------------------------
# FILE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # -----------------------------------
    # LOAD DATA
    # -----------------------------------
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # EDA SECTION
    # -----------------------------------
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Column Names")
    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(missing_df)

    st.subheader("Data Types")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str)
        })
    )

    # -----------------------------------
    # PREPROCESSING
    # -----------------------------------
    processed_df = df.copy()

    # Remove completely empty columns
    processed_df.dropna(
        axis=1,
        how="all",
        inplace=True
    )

    # Handle datetime columns
    for col in processed_df.columns:

        try:
            converted = pd.to_datetime(
                processed_df[col],
                errors="coerce"
            )

            valid_ratio = converted.notna().mean()

            if valid_ratio > 0.8:
                processed_df[col] = (
                    converted.astype("int64")
                    // 10**9
                )

        except:
            pass

    # Encode categorical columns
    for col in processed_df.columns:

        if processed_df[col].dtype == "object":

            processed_df[col] = (
                processed_df[col]
                .astype(str)
                .fillna("Unknown")
            )

            le = LabelEncoder()

            processed_df[col] = le.fit_transform(
                processed_df[col]
            )

    # Convert everything to numeric
    processed_df = processed_df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Fill missing values
    processed_df.fillna(0, inplace=True)

    # Keep numeric columns only
    processed_df = processed_df.select_dtypes(
        include=np.number
    )

    if processed_df.shape[1] < 2:
        st.error(
            "Dataset must contain at least 2 usable numerical columns."
        )
        st.stop()

    # -----------------------------------
    # STATISTICS
    # -----------------------------------
    st.subheader("Statistical Summary")
    st.dataframe(processed_df.describe())

    # -----------------------------------
    # SCALING
    # -----------------------------------
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(
        processed_df
    )

    scaled_df = pd.DataFrame(
        scaled_data,
        columns=processed_df.columns
    )

    # -----------------------------------
    # CORRELATION HEATMAP
    # -----------------------------------
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        scaled_df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # -----------------------------------
    # DISTRIBUTION PLOT
    # -----------------------------------
    st.subheader("Feature Distribution")

    selected_col = st.selectbox(
        "Select Feature",
        scaled_df.columns
    )

    fig, ax = plt.subplots()

    sns.histplot(
        scaled_df[selected_col],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # -----------------------------------
    # DENDROGRAM
    # -----------------------------------
    st.header("Hierarchical Clustering")

    linkage_method = st.selectbox(
        "Linkage Method",
        ["ward", "complete", "average", "single"]
    )

    linkage_matrix = linkage(
        scaled_df,
        method=linkage_method
    )

    st.subheader("Dendrogram")

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    dendrogram(
        linkage_matrix,
        truncate_mode="level",
        p=5,
        ax=ax
    )

    ax.set_title(
        "Hierarchical Clustering Dendrogram"
    )

    st.pyplot(fig)

    # -----------------------------------
    # CLUSTERING
    # -----------------------------------
    n_clusters = st.slider(
        "Select Number of Clusters",
        min_value=2,
        max_value=10,
        value=3
    )

    model = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    clusters = model.fit_predict(
        scaled_df
    )

    result_df = df.copy()

    result_df["Cluster"] = clusters

    # -----------------------------------
    # RESULTS
    # -----------------------------------
    st.subheader("Clustered Dataset")

    st.dataframe(result_df)

    st.subheader("Cluster Distribution")

    st.bar_chart(
        result_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    # -----------------------------------
    # DOWNLOAD
    # -----------------------------------
    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Clustered Dataset",
        csv,
        "hierarchical_clustering_output.csv",
        "text/csv"
    )

else:
    st.info(
        "Please upload a CSV file to begin."
    )
