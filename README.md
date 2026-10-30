
# Wine Dataset Clustering Project

## About
This project uses unsupervised machine learning to group wine samples into clusters.

## Dataset
- 178 samples
- 13 features
- 0 missing values

## Technologies
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Algorithms
- K-Means Clustering
- Hierarchical Clustering
- Principal Component Analysis (PCA)

## Results
| Method | Silhouette Score |
|---|---:|
| K-Means | 0.285 |
| Hierarchical | 0.277 |

### Cluster Count Comparison
| Clusters | Score |
|---|---:|
| 2 | 0.259 |
| 3 | 0.285 |
| 4 | 0.260 |
| 5 | 0.202 |

PCA retained 55.4% of the total variance using two components.

## Project Files
- main.py
- cluster_results.csv
- model_evaluation.csv
- clusters.png
- pca_clusters.png
