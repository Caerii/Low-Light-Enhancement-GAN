import pandas as pd
import colour 
import numpy as np
import statistics
from sklearn.cluster import KMeans
import random
from sklearn.decomposition import PCA
df = pd.read_csv(r'C:\Users\Josep\OneDrive\Desktop\DTU course\day1\Coloimetry\Munsell\xyz_m.csv')
x= np.zeros(1286)
x= df['x']
x_mean =statistics.mean(x)
y = np.zeros(1286)
y=df['y']
y_mean = statistics.mean (y)
z = np.zeros(1286)
z = df['z']
z_mean = statistics.mean(z)
print(x_mean,y_mean,z_mean)
