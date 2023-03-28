import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


### K-Means
### Write a program to use k-means clustering on the Madelon dataset.
### Try the following k values: 4, 8, 16, 32, 64. 

### Data Preprocessing

# read the data
madelon = pd.read_csv('Madelon.csv')
madelon.head(10)

# remove the index column in the original data set
madelon = madelon[madelon.columns[1:]]

# scale the data
scale_madelon = pd.DataFrame(preprocessing.scale(madelon, axis = 0))
scale_madelon.describe()

# using scaled data
sse = []
k_range = [4, 8, 16, 32, 64]
for k in k_range:
    kmeans = KMeans(n_clusters = k, random_state = 0)
    kmeans.fit(scale_madelon)
    sse.append(kmeans.inertia_)
    
# generate elbow plot
plt.plot(k_range, sse, marker = '*', markersize = 8)
plt.title('No of clusters k versus SSE')
plt.xlabel('Number of clusters k')
plt.ylabel('sum of squared distances (SSE)')
plt.show()

# use elbow method to determine
kl = KneeLocator(k_range, sse, curve = 'convex', direction = 'decreasing')
kl.elbow

### init = k-means++ vs random

k = 8
# init = 'kmeans++'
kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 0)
kmeans.fit(scale_madelon)
centroids_kmeans = kmeans.cluster_centers_

# init = 'random'
kmeans_random = KMeans(n_clusters = k, init = 'random', random_state = 0)
kmeans_random.fit(scale_madelon)
centroids_random = kmeans_random.cluster_centers_

# make sure the centroids are all different
comparison = centroids_kmeans == centroids_random
equal_centroids = comparison.all()
print('Whether two kmeans algorithm contain same centroids:', equal_centroids)

# compare the sse
print('SSE for init = k-means++:', kmeans.inertia_)
print('SSE for init = random:', kmeans_random.inertia_)
