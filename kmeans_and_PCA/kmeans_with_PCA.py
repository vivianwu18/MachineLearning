import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


### K-Means with PCA
### 1. Before performing any dimensionality reduction, write a program to use k-means clustering on the Madelon dataset.
### - Try the following k values: 4, 8, 16, 32, 64. 

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


### 2. The Madelon dataset is high-dimensional, with 500 features per data point. Some of these features might be redundant or noisy, making clustering more difficult. 
### - Fit the standardized data with PCA. Then, create a cumulative variance plot – showing the number of components included (x-axis) versus the amount of variance captured (y-axis).
### Generally, we want to retain at least 75% of the variance. How many components would you decide to keep?


# Perform PCA - method 1
pca1 = PCA(n_components = 500)
pca1.fit(scale_madelon)
list = pca1.explained_variance_ratio_

varianceRatio = []
j = 0
for i in range(0, len(list)):
    j += list[i]
    varianceRatio.append(j)
    
# find the no of components which we retain at least 75% of the variance
for i in range(1, 500):
    if varianceRatio[i] >= 0.75:
        print('The number of components should be', i + 1, '.')
        break

# Perform PCA - method 2
pca2 = PCA(n_components = 0.75)
pca2.fit(scale_madelon)
print('The number of components should be', pca2.n_components_, '.')

# plot PCA
plt.plot(range(1, 501), varianceRatio)
plt.plot(277, 0.75, 'ro', markersize = 5)
plt.axhline(y = 0.75, color = 'gray', linestyle = '--')
plt.title('No of clusters k versus SSE')
plt.xlabel('Number of clusters k')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.text(280, 0.65, '277 components')


### 3. Perform PCA with selected principal components.
### - Plot the transformed data on a graph with the first two principal components as the axes i.e. x = PC 1, y = PC 2.


# perform PCA 
pca_final = PCA(n_components = 277, random_state = 0)
pca_final.fit(scale_madelon)

transformed_madelon = pd.DataFrame(pca_final.transform(scale_madelon))
transformed_madelon

# plot the transformed data with the first two principal components
plt.scatter(transformed_madelon.iloc[:, 0], transformed_madelon.iloc[:, 1], alpha = 0.5)
plt.show()


### - Plot the original data on a graph with the two original variables that have the highest absolute combined loading for PC 1 and PC 2 i.e. maximizing |loading PC1| + |loading PC2|.


# find out the variables having the highest absolute combined loading
loading = pca_final.components_.T
loading = pd.DataFrame(loading)
loading['value'] = abs(loading[0]) + abs(loading[1])
loading['value'].sort_values(ascending = False)

# generate the plot
plt.scatter(madelon.iloc[:, 338], madelon.iloc[:, 281], alpha = 0.5)
plt.xlabel('Variable 339')
plt.ylabel('Variable 282')
plt.show()


### 4 Use the same k values again (4, 8, 16, 32, 64) to again generate an elbow plot. 
### - What is the optimal k? Is it different from the one you found in (1)?


sse_trans = []
k_range = [4, 8, 16, 32, 64]
for k in k_range:
    kmeans = KMeans(n_clusters = k, random_state = 0)
    kmeans.fit(transformed_madelon)
    sse_trans.append(kmeans.inertia_)
    
print(sse_trans)

# generate elbow plot
plt.figure(figsize = (18, 5))

plt.subplot(1, 2, 2)
plt.plot(k_range, sse_trans, marker = '*', markersize = 8)
plt.title('No of clusters k versus SSE after PCA')
plt.xlabel('Number of clusters k')
plt.ylabel('sum of squared distances (SSE)')


# use elbow method to determine
k2 = KneeLocator(k_range, sse_trans, curve = 'convex', direction = 'decreasing')

k2.elbow


# compare two elbow plots with/ without PCA
plt.figure(figsize = (18, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, sse, marker = '*', markersize = 8)
plt.title('No of clusters k versus SSE before PCA')
plt.xlabel('Number of clusters k')
plt.ylabel('sum of squared distances (SSE)')

plt.subplot(1, 2, 2)
plt.plot(k_range, sse_trans, marker = '*', markersize = 8)
plt.title('No of clusters k versus SSE after PCA')
plt.xlabel('Number of clusters k')
plt.ylabel('sum of squared distances (SSE)')

# create the scatter plot based on different iterations
k = 32
for i in range(1, 6):
    kmeans = KMeans(n_clusters = k, random_state = 0, max_iter = i)
    kmeans.fit(transformed_madelon)
    plt.scatter(transformed_madelon.iloc[:, 0], transformed_madelon.iloc[:, 1], c = kmeans.labels_, alpha = 0.5)
    plt.scatter(pd.DataFrame(kmeans.cluster_centers_).iloc[:, 0], pd.DataFrame(kmeans.cluster_centers_).iloc[:, 1], 
                c = 'black', marker = 'X', s = 50)
    plt.title(f'Iterations = {i}')
    plt.show()
