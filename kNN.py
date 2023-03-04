import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from math import sqrt


### KNN
### 1. Use the iris.csv dataset for this question. How does the performance of k-nearest neighbors change as k takes on the following values: 1, 3, 5, 7? Which of these is the optimal value of k? Which distance/similarity metric did you choose to use and why?


# read the data set
iris = pd.read_csv('iris.csv')
iris.head(10)

# know how many categories we have
iris['variety'].unique()

# split the data into training and testing parts
x = iris.iloc[:, : -1]
y = iris.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.2)

# try different k in knn algorithm
k_list = [1, 3, 5, 7]
distance_list = ['euclidean', 'manhattan', 'minkowski']
result = pd.DataFrame(columns = distance_list)

for i in range(len(distance_list)):
    distance = distance_list[i]
    accuracy_list = []
    for j in range(len(k_list)):
        k = k_list[j]
        knn = KNeighborsClassifier(n_neighbors = k, metric = distance)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_list.append(accuracy)
    result.iloc[:, i] = accuracy_list

result['k'] = k_list
print(result)

# plot the result
plt.plot('k', 'euclidean', data = result, label = 'euclidean')
plt.plot('k', 'manhattan', data = result, label = 'manhattan')
plt.plot('k', 'minkowski', data = result, label = 'minkowski')
plt.xlabel('k neighbors (1, 3, 5, 7)')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.show()
