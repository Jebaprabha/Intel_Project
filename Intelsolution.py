import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.iloc[:, :-1])

linked = linkage(scaled_data, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=df['species'].to_numpy(), leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points (Iris Species)')
plt.ylabel('Distance')
plt.show()

cluster = AgglomerativeClustering(n_clusters=3)
df['cluster'] = cluster.fit_predict(scaled_data)

score = silhouette_score(scaled_data, df['cluster'])
print(f'Silhouette Score: {score:.4f}')

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['sepal length (cm)'], y=df['sepal width (cm)'], hue=df['cluster'], palette='viridis')
plt.title('Clusters of Iris Dataset (Agglomerative Clustering)')
plt.show()
