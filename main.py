
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# Load dataset
data = load_wine(as_frame=True).data

print("Dataset:", data.shape)
print("Missing values:", data.isnull().sum().sum())

# Scale data
X = StandardScaler().fit_transform(data)

# PCA: reduce 13 features to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("PCA variance:", round(sum(pca.explained_variance_ratio_), 3))

# Test different cluster counts
results = []

for n in [2, 3, 4, 5]:
    model = KMeans(n_clusters=n, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)

    results.append({
        "Clusters": n,
        "Silhouette Score": round(score, 3)
    })

evaluation = pd.DataFrame(results)
print(evaluation)

# Final K-Means model
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
h_labels = hierarchical.fit_predict(X)

print("K-Means:", round(silhouette_score(X, k_labels), 3))
print("Hierarchical:", round(silhouette_score(X, h_labels), 3))

# Save results
data["KMeans"] = k_labels
data["Hierarchical"] = h_labels

data.to_csv("cluster_results.csv", index=False)
evaluation.to_csv("model_evaluation.csv", index=False)

# PCA graph
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels)
plt.title("Wine Clusters using PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig("pca_clusters.png")
plt.show()

print("All files saved successfully!")
