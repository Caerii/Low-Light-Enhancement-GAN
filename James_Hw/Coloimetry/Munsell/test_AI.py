# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import colour
# Unlabled training data
xyzdata = pd.read_csv('xyz_m.csv')
dataPrep = list(zip(xyzdata['x'],xyzdata['y'],xyzdata['z']))
dataPrepX= colour.XYZ_to_Lab(dataPrep)
data = np.array(dataPrepX)
# data = np.array([[19, 15, 39], [67, 19, 14], [35, 24, 35], [60, 30, 4], [65, 38, 35],[49, 42, 52], [70, 46, 56], [70, 49, 55], [57, 54, 51], [68, 59, 55],[23, 62, 41], [65, 63, 52], [27, 67, 56], [47, 71, 9],  [57, 75, 5],[43, 78, 17], [56, 79, 35], [40, 87, 13], [37, 97, 32], [34, 103, 23]])
# Initial centroids
# init_centroids = np.random.uniform(low=0.0, high=1.0, size=(1269,3))
init_centroids = np.random.uniform(low=0.0, high=1.0, size=(3,3))
# init_centroids = np.array([[70, 46, 56], [27, 67, 56], [37, 97, 32]])
# Create a KMeans object by specifying
# - Number of clusters (n_clusters) = 3, initial centroids (init) = init_centroids
# - Number of time the k-means algorithm will be run with different centroid seeds (n_init) = 1
# - Maximum number of iterations of the k-means algorithm for a single run (max_iter) = 4
kmeans = KMeans(n_clusters=3, init=init_centroids, n_init=1, max_iter = 4)
kmeans.fit(data) # Compute k-means clustering

labels = kmeans.predict(data)# Predict the closest cluster each sample in data belongs to
centroids = kmeans.cluster_centers_# Get resulting centroids
fig = plt.figure(figsize = (10,10))# Figure width = 10 inches, height = 10 inches
ax = fig.gca(projection='3d')# Defining 3D axes so that we can plot 3D data into it
# Get boolean arrays representing entries with labels = 0, 1, and 2
a = np.array(labels == 0); b = np.array(labels == 1); c = np.array(labels == 2)
# Plot centroids with color = black, size = 50 units, transparency = 20%, and put label "Centroids"
ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2],c="black", s=50, alpha=0.8, label="Centroids")
# Plot data in the different clusters (1st in red, 2nd in green, 3rd blue)
ax.scatter(data[a,0], data[a,1], data[a,2], c="red", s=40, label="1st Cluster")
ax.scatter(data[b,0], data[b,1], data[b,2], c="green", s=40, label="2nd Cluster")
ax.scatter(data[c,0], data[c,1], data[c,2], c="blue", s=40, label="3rd Cluster")
ax.legend()# Show legend
# ax.set_xlabel("Age")# Put x-axis label "Age"
# ax.set_ylabel("Income (K)")# Put y-axis label "Income (K)"
# ax.set_zlabel("Expense Score (1-100)")# Put z-axis label "Expense Score (1-100)"
ax.set_xlabel("x")# Put x-axis label "Age"
ax.set_ylabel("y")# Put y-axis label "Income (K)"
ax.set_zlabel("z")# Put z-axis label "Expense Score (1-100)"
ax.set_title("Customer Segmentation - K-Means Clustering")# Put figure title
plt.show()

# https://towardsdatascience.com/10-tips-for-choosing-the-optimal-number-of-clusters-277e93d72d92
# silhouette-optimal number of clusters
# 
