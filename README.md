# Intel_Project

Iris Dataset Clustering using Hierarchical Agglomerative Clustering

Project Overview:
This project focuses on performing Hierarchical Agglomerative Clustering on the famous Iris dataset. The goal is to group the iris flowers into distinct clusters based on their sepal and petal measurements and evaluate the clustering quality using the Silhouette Score.

Problem Statement:
The Iris dataset contains measurements of iris flowers' sepals and petals across three species: Setosa, Versicolor, and Virginica. The objective is to apply unsupervised clustering techniques to:
Identify natural groupings of iris flowers without prior labels.
Visualize the clustering process using dendrograms.
Evaluate the clustering performance through the silhouette score.

Technologies Used:
Python
Pandas
Matplotlib
Seaborn
Scikit-learn
SciPy

Dataset:
The Iris dataset is loaded from Scikit-learn’s built-in datasets. It contains:
150 samples (50 from each species)

4 features:
Sepal length (cm)
Sepal width (cm)
Petal length (cm)
Petal width (cm)
Project Structure

Data Preprocessing:
Load the Iris dataset.
Standardize the features using StandardScaler.

Hierarchical Clustering:
Perform agglomerative clustering using the Ward method.
Visualize the clustering hierarchy with a dendrogram.

Clustering and Evaluation:
Cluster the data into 3 groups.
Calculate the silhouette score to measure clustering quality.

Visualization:
Plot the clusters with a scatter plot.

Results:
Dendrogram: Displays the hierarchical clustering process.
Silhouette Score: Quantifies the cohesion and separation of clusters.
Scatter Plot: Visualizes the resulting clusters in a 2D space.
