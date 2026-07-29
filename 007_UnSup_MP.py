import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage


# ==========================================
# Create Sample Dataset (No Target Column)
# ==========================================

data = {
    "Age": [22,25,45,50,23,48,30,35,60,62,28,32],
    "Income": [30000,35000,90000,95000,32000,88000,
               50000,52000,100000,105000,45000,48000],
    "SpendingScore": [80,85,20,25,82,22,
                      60,58,15,18,65,63]
}

df = pd.DataFrame(data)

print(df)


# ==========================================
# Feature Selection
# ==========================================

X = df


# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ==========================================
# Elbow Method
# ==========================================

inertias = []

for k in range(1,8):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    inertias.append(model.inertia_)

plt.figure(figsize=(6,4))

plt.plot(range(1,8), inertias, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()


# ==========================================
# K-Means Clustering
# ==========================================

input_k = input("Enter the number of clusters (K) for K-Means: ")

kmeans = KMeans(
    n_clusters=int(input_k),
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

df["KMeans Cluster"] = kmeans_labels

print("\nK-Means Clusters")

print(df)

print("\nCentroids")

print(kmeans.cluster_centers_)

print("\nInertia")

print(kmeans.inertia_)


# ==========================================
# Hierarchical Clustering
# ==========================================

linked = linkage(
    X_scaled,
    method="ward"
)

plt.figure(figsize=(8,5))

dendrogram(linked)

plt.title("Dendrogram")

plt.xlabel("Customers")

plt.ylabel("Distance")

plt.show()


# ==========================================
# Agglomerative Clustering
# ==========================================

agg = AgglomerativeClustering(
    n_clusters=3
)

agg_labels = agg.fit_predict(X_scaled)

df["Agglomerative Cluster"] = agg_labels

print("\nAgglomerative Clusters")

print(df)


# ==========================================
# DBSCAN Clustering
# ==========================================

dbscan = DBSCAN(
    eps=1.2,
    min_samples=2
)

dbscan_labels = dbscan.fit_predict(X_scaled)

df["DBSCAN Cluster"] = dbscan_labels

print("\nDBSCAN Clusters")

print(df)

print("\nDBSCAN Labels")

print(dbscan_labels)

print("\nNote:")
print("-1 means Noise / Outlier")


# ==========================================
# PCA
# ==========================================

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPrincipal Components")

print(X_pca)

print("\nExplained Variance Ratio")

print(pca.explained_variance_ratio_)


plt.figure(figsize=(6,5))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=kmeans_labels
)

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.title("PCA")

plt.show()


# ==========================================
# t-SNE
# ==========================================

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=5
)

X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(6,5))

plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=kmeans_labels
)

plt.title("t-SNE")

plt.show()


# ==========================================
# Silhouette Score
# ==========================================

score = silhouette_score(
    X_scaled,
    kmeans_labels
)

print("\nSilhouette Score")

print(score)


# ==========================================
# Cluster Interpretation
# ==========================================

print("\nCluster Interpretation")

print(df.groupby("KMeans Cluster").mean())