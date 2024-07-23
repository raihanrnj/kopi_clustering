import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from PIL import Image
import pickle
from sklearn_extra.cluster import KMedoids

# Fungsi untuk memuat data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Fungsi untuk preprocessing data
def preprocess_data(df):
    df = df[['Provinsi', 'Kabupaten/Kota','TON_MEAN', 'TON_SUM','L_MEAN', 'L_SUM']]
    columns_to_convert = ['TON_MEAN', 'TON_SUM', 'L_MEAN', 'L_SUM']
    
    for col in columns_to_convert:
        df[col] = df[col].replace('#VALUE!', np.nan)  # Mengganti #VALUE! dengan NaN  
        df[col] = df[col].replace('#DIV/0!', np.nan)  # Mengganti #VALUE! dengan NaN
        df[col] = df[col].str.replace(',', '.')       # Mengganti koma dengan titik
        df[col] = df[col].astype(float).fillna(0)

    df = df[(df['L_MEAN'] > 0) & (df['L_SUM'] > 0)]
    df = df[~df['Kabupaten/Kota'].isin(['OKU Selatan', 'Lahat'])]
    
    return df

# # Fungsi untuk menampilkan clustering hasil
# def plot_clustering(X, labels, title):
#     plt.figure(figsize=(10, 7))
#     plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
#     plt.title(title)
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     st.pyplot()
# Fungsi untuk menampilkan clustering hasil
# Fungsi untuk menampilkan clustering hasil
def plot_clustering(X, labels, title, centroids=None):
    plt.figure(figsize=(10, 7))
    # Jika X adalah DataFrame, ubah menjadi numpy array
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    if centroids is not None:
        # Jika centroids adalah DataFrame, ubah menjadi numpy array
        if isinstance(centroids, pd.DataFrame):
            centroids = centroids.to_numpy()
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot()


# Fungsi untuk KMeans clustering
def kmeans_clustering(df, X):
    st.write("## KMeans Clustering")
    min_clusters = 2
    max_clusters = 11
    best_silhouette_score = 0
    best_clusters = None

    for n_clusters in range(min_clusters, max_clusters):
        clusterer = KMeans(n_clusters=n_clusters)
        preds = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, preds)
        dbi_score = davies_bouldin_score(X, preds)

        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_clusters = n_clusters

        st.write(f"Jumlah Cluster = {n_clusters}, Silhouette Score = {silhouette_avg}, Davies-Bouldin Index = {dbi_score}")

    st.write(f"Cluster terbaik berdasarkan Silhouette Score: {best_clusters} dengan nilai Silhouette Score: {best_silhouette_score}")

    # Plot elbow method
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    st.pyplot()

    kmeans = KMeans(n_clusters=best_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    plot_clustering(X, df['Cluster'], 'KMeans Clustering', centroids)

    # Display cluster counts
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("Jumlah data dalam masing-masing cluster:")
    st.write(cluster_counts)

    # Display clusters
    for cluster in range(best_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"\nCluster {cluster}:")
        st.write(cluster_data[['Kabupaten/Kota']])

    # Display silhouette score
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"\nSilhouette Score: {silhouette_avg}")

# Fungsi untuk Agglomerative Clustering
def agglomerative_clustering(df, X):
    st.write("## Agglomerative Clustering")
    dbi_scores = []
    silhouette_scores = []

    for n_clusters in range(2, 11):
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        df['Cluster'] = agg_clustering.fit_predict(X)

        dbi = davies_bouldin_score(X, df['Cluster'])
        silhouette = silhouette_score(X, df['Cluster'])

        dbi_scores.append(dbi)
        silhouette_scores.append(silhouette)

        st.write(f"Jumlah Cluster = {n_clusters}, Silhouette Score = {silhouette}, Davies-Bouldin Index = {dbi}")

    # Plotting DBI
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 11), dbi_scores, marker='o')
    plt.title('Davies-Bouldin Index')
    plt.xlabel('Number of Clusters')
    plt.ylabel('DBI Score')
    plt.grid(True)
    st.pyplot()

    # Plotting Silhouette
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    st.pyplot()

    optimal_clusters = 2  # Change this based on the best result
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters)
    df['Cluster'] = agg_clustering.fit_predict(X)
    
    plot_clustering(X, df['Cluster'], 'Agglomerative Clustering')

    # Display cluster counts
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("Jumlah data dalam masing-masing cluster:")
    st.write(cluster_counts)

    # Display clusters
    for cluster in range(optimal_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"\nCluster {cluster}:")
        st.write(cluster_data[['Kabupaten/Kota']])

    # Display silhouette score
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"\nSilhouette Score: {silhouette_avg}")

# Fungsi untuk DBSCAN clustering
def dbscan_clustering(df, X):
    st.write("## DBSCAN Clustering")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['Cluster'] = dbscan.fit_predict(X)

    plot_clustering(X, df['Cluster'], 'DBSCAN Clustering')

    # Display cluster counts
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("Jumlah data dalam masing-masing cluster:")
    st.write(cluster_counts)

    # Display clusters
    for cluster in np.unique(df['Cluster']):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"\nCluster {cluster}:")
        st.write(cluster_data[['Kabupaten/Kota']])

    # Display silhouette score
    if len(set(df['Cluster'])) > 1:
        silhouette_avg = silhouette_score(X, df['Cluster'])
        st.write(f"\nSilhouette Score: {silhouette_avg}")
    else:
        st.write("\nSilhouette Score: Not applicable (only one cluster)")

# Fungsi untuk Gaussian Mixture Model Clustering
def gmm_clustering(df, X):
    st.write("## Gaussian Mixture Model Clustering")
    bic_scores = []

    min_clusters = 2
    max_clusters = 10

    for n_clusters in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        labels = gmm.predict(X)
        silhouette_avg = silhouette_score(X, labels)
        dbi_score = davies_bouldin_score(X, labels)
        st.write(f"Jumlah Cluster = {n_clusters}, BIC = {bic_scores[-1]}, Silhouette Score = {silhouette_avg}, Davies-Bouldin Index = {dbi_score}")

    optimal_clusters = np.argmin(bic_scores) + min_clusters
    st.write(f"Optimal number of clusters based on BIC: {optimal_clusters}")

    gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
    df['Cluster'] = gmm.fit_predict(X)
    centroids = gmm.means_

    plot_clustering(X, df['Cluster'], 'Gaussian Mixture Model Clustering', centroids)

    # Display cluster counts
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("Jumlah data dalam masing-masing cluster:")
    st.write(cluster_counts)

    # Display clusters
    for cluster in range(optimal_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"\nCluster {cluster}:")
        st.write(cluster_data[['Kabupaten/Kota']])

    # Display silhouette score
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"\nSilhouette Score: {silhouette_avg}")
    
# Fungsi untuk KMedoids clustering
def kmedoids_clustering(df, X):
    st.write("## KMedoids Clustering")
    min_clusters = 2
    max_clusters = 10
    best_silhouette_score = 0
    best_clusters = None

    for n_clusters in range(min_clusters, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
        preds = kmedoids.fit_predict(X)
        
        silhouette_avg = silhouette_score(X, preds)
        dbi_score = davies_bouldin_score(X, preds)
        
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_clusters = n_clusters
            
        st.write(f"Jumlah Cluster = {n_clusters}, Silhouette Score = {silhouette_avg}, Davies-Bouldin Index = {dbi_score}")
    
    st.write(f"Cluster terbaik berdasarkan Silhouette Score: {best_clusters} dengan nilai Silhouette Score: {best_silhouette_score}")

    kmedoids = KMedoids(n_clusters=best_clusters, random_state=42)
    df['Cluster'] = kmedoids.fit_predict(X)
    centroids = kmedoids.cluster_centers_

    # Ensure centroids is a valid NumPy array
    if isinstance(centroids, np.ndarray) and centroids.ndim == 2:
        plot_clustering(X, df['Cluster'], 'KMedoids Clustering', centroids)
    else:
        st.write("Error: Centroids array is not valid or not in the correct format.")

    # Display cluster counts
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.write("Jumlah data dalam masing-masing cluster:")
    st.write(cluster_counts)

    # Display clusters
    for cluster in range(best_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        st.write(f"\nCluster {cluster}:")
        st.write(cluster_data[['Kabupaten/Kota']])

    # Display silhouette score
    silhouette_avg = silhouette_score(X, df['Cluster'])
    st.write(f"\nSilhouette Score: {silhouette_avg}")
    
# Fungsi untuk memuat data
def load_data(file):
    df = pd.read_csv(file, sep=",")
    return df

# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    # Tambahkan ikon dan judul utama
    coffee_icon = Image.open('kopi.png')  # Pastikan kamu memiliki file coffee_icon.png
    st.image(coffee_icon, width=60)
    st.title('CoffeeZones')  # Menambahkan judul di Streamlit
    st.title("Analisis Klusterisasi Kopi di Kabupaten/Kota")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.sidebar.title("CoffeeZones")
    st.sidebar.title("Menu")
    menu = ["Overview", "Upload Data Training", "Data Preprocessing", "Modelling & Evaluation", "Upload & Predict"]
    choice = st.sidebar.selectbox("Pilih Menu", menu)

    if choice == "Overview":
        st.subheader("Overview")
        st.write("""
        Analisis clustering adalah teknik yang digunakan untuk mengelompokkan data ke dalam cluster yang memiliki kesamaan. 
        Dalam aplikasi ini, kami akan menggunakan berbagai metode clustering untuk menganalisis data Kabupaten/Kota.
        """)

    elif choice == "Upload Data Training":
        st.subheader("Upload Data Training")
        uploaded_file = st.file_uploader("Unggah file CSV data", type=["csv"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.write("Data berhasil diunggah!")
            st.write(df.head())
            df.to_csv('data.csv', index=False)

    elif choice == "Data Preprocessing":
        st.subheader("Data Preprocessing")
        df = load_data('data.csv')
        df = preprocess_data(df)
        st.write("Data berhasil diproses!")
        st.write(df.head())
        df.to_csv('preprocessed_data.csv', index=False)

    elif choice == "Modelling & Evaluation":
        st.subheader("Modelling & Evaluation")
        df = load_data('preprocessed_data.csv')
        # X = df[['TON_MEAN', 'TON_SUM', 'L_MEAN', 'L_SUM']]
        # The line `X = df[['TON_MEAN', 'TON_SUM', 'L_MEAN', 'L_SUM']]` is selecting specific columns
        # from the DataFrame `df` and assigning them to a new DataFrame `X`.
        X = df[['TON_SUM', 'L_MEAN']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        

        best_algo = ""
        best_score = -1

        st.write("Menjalankan semua metode clustering...")

        # KMeans Clustering
        min_clusters = 2
        max_clusters = 11
        best_silhouette_score = 0
        best_clusters = None
        for n_clusters in range(min_clusters, max_clusters):
            clusterer = KMeans(n_clusters=n_clusters)
            preds = clusterer.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, preds)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_clusters = n_clusters
        kmeans = KMeans(n_clusters=best_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        kmeans_silhouette = silhouette_score(X_scaled, df['Cluster'])
        if kmeans_silhouette > best_score:
            best_score = kmeans_silhouette
            best_algo = "KMeans"

        # Agglomerative Clustering
        best_silhouette_score = 0
        for n_clusters in range(2, 11):
            agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
            df['Cluster'] = agg_clustering.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        df['Cluster'] = agg_clustering.fit_predict(X_scaled)
        agg_silhouette = silhouette_score(X_scaled, df['Cluster'])
        if agg_silhouette > best_score:
            best_score = agg_silhouette
            best_algo = "Agglomerative Clustering"

        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        df['Cluster'] = dbscan.fit_predict(X_scaled)
        if len(set(df['Cluster'])) > 1:
            dbscan_silhouette = silhouette_score(X_scaled, df['Cluster'])
            if dbscan_silhouette > best_score:
                best_score = dbscan_silhouette
                best_algo = "DBSCAN"
        else:
            dbscan_silhouette = -1

        # Gaussian Mixture Model Clustering
        bic_scores = []
        min_clusters = 2
        max_clusters = 10
        best_silhouette_score = 0
        for n_clusters in range(min_clusters, max_clusters + 1):
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, labels)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        df['Cluster'] = gmm.fit_predict(X_scaled)
        gmm_silhouette = silhouette_score(X_scaled, df['Cluster'])
        if gmm_silhouette > best_score:
            best_score = gmm_silhouette
            best_algo = "Gaussian Mixture Model"
            
        # KMedoids Clustering
        best_silhouette_score = 0
        for n_clusters in range(min_clusters, max_clusters):
            kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
            preds = kmedoids.fit_predict(X_scaled)
            silhouette_avg = silhouette_score(X_scaled, preds)
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_clusters = n_clusters
        kmedoids = KMedoids(n_clusters=best_clusters, random_state=42)
        df['Cluster'] = kmedoids.fit_predict(X_scaled)
        kmedoids_silhouette = silhouette_score(X_scaled, df['Cluster'])
        if kmedoids_silhouette > best_score:
            best_score = kmedoids_silhouette
            best_algo = "KMedoids"

        # Teks yang ingin diubah
        text = f"Algoritma terbaik adalah {best_algo} dengan kluster = 2 dan Silhouette Score: {best_score}"

        # Definisi CSS untuk teks dengan border biru
        css = """
        <style>
        .highlight-box {
            display: inline-block;
            padding: 10px;
            border: 2px solid blue; /* Warna border */
            border-radius: 5px; /* Radius border */
            background-color: #e0f7fa; /* Warna background (biru muda) */
            color: black; /* Warna teks */
            font-size: 18px; /* Ukuran font */
        }
        </style>
        """

        # Tampilkan CSS dan teks dengan bounding box di Streamlit
        st.markdown(css, unsafe_allow_html=True)
        st.markdown(f'<div class="highlight-box">{text}</div>', unsafe_allow_html=True)
        
        # Menampilkan algoritma terbaik
        # st.write(f"Algoritma terbaik adalah {best_algo} dengan kluster = 2 dan Silhouette Score: {best_score}")


        kmeans_clustering(df, X_scaled)
        agglomerative_clustering(df, X_scaled)
        dbscan_clustering(df, X_scaled)
        gmm_clustering(df, X_scaled)
        kmedoids_clustering(df, X)
        
        
    elif choice == "Upload & Predict":
        st.subheader("Upload & Predict using KMEANS with 3 Cluster")
        uploaded_file = st.file_uploader("Unggah file CSV untuk prediksi", type=["csv"])
        if uploaded_file is not None:
            # df_test = load_data(uploaded_file)
            df_test = pd.read_csv(uploaded_file, sep=";")
            st.write("Data untuk prediksi berhasil diunggah!")
            st.write(df_test.head())
            
            df_test = preprocess_data(df_test)
            X_test = df_test[['TON_SUM', 'L_MEAN']]
            X_test_scaled = StandardScaler().fit_transform(X_test)
            
            # Load trained KMeans model
            with open('kmeans_model.pkl', 'rb') as file:
                loaded_kmeans = pickle.load(file)
            
            df_test['Cluster'] = loaded_kmeans.predict(X_test_scaled)
            
            st.write("Hasil Prediksi Cluster:")
            st.write(df_test)
            
            plot_clustering(X_test_scaled, df_test['Cluster'], 'KMeans Clustering for Testing Data')
            
            # Button to export CSV
            csv = df_test.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download hasil prediksi sebagai CSV",
                data=csv,
                file_name='hasil_prediksi.csv',
                mime='text/csv',
            )

    

if __name__ == "__main__":
    main()
