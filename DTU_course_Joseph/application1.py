import pandas as pd
import colour 
import numpy as np
from sklearn.cluster import KMeans
import random
df = pd.read_csv(r'C:\Users\Josep\OneDrive\Desktop\DTU course\Coloimetry\Munsell\xyz_m.csv')
XYZ = np.zeros((1268, 3))
Lab = np.zeros((1268, 3))
sample1_xy = np. zeros ((1268,2))
for i in range(1268):
    XYZ[i] = np.array(df.iloc[i, 1:])
    Lab[i] = colour.XYZ_to_Lab(XYZ[i])
i = random.randint(0,1268)
j = random.randint(0,1268)
delta_e = colour.delta_E(Lab[i],Lab[j])
print (delta_e)
kmeans = KMeans(n_clusters=5)
kmeans.fit(Lab)

# Get the cluster labels for each color
labels = kmeans.labels_

# Print the result
for i in range(5):
    cluster_colors = Lab[labels == i]
    print(f"Cluster {i+1}: {len(cluster_colors)} colors")
    print(f"Mean color: {colour.utilities.tstack(np.mean(cluster_colors, axis=0))}\n")

