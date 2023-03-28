import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.formula.api as smf


### 1. Use the drinking.csv dataset for this question.
### Keeping 21 as the threshold for age, explore the data with an RDD by writing very simple code (no package needed, just average to one side of the threshold minus average to the other side) to determine if alcohol increases the chances of death by accident, suicide and/or others (the three given columns) and comment on the question “Should the legal age for drinking be reduced from 21?” based on the results. Plot graphs to show the discontinuity (if any) and to show results for the change in chances of death with all the three features (i.e., accident vs age, suicide vs age and others vs age). For this problem, choose the bandwidth to be 1 year (i.e., 21 +- 1). What might be the effect of choosing a smaller bandwidth?  What if we chose the maximum bandwidth?

# In[10]:


# read the data set
drinking = pd.read_csv('drinking.csv')
drinking.head(10)

# check for missing data
drinking.isnull().mean()

# remove invalid data
drinking = drinking.dropna()
drinking.isnull().mean()

# group the data with the threshold for age 21 and compute the average
drinking = drinking[(drinking['age'] <= 22) & (drinking['age'] >= 20)]
drinking['Group'] = np.where(drinking['age'] < 21, 0, 1)
drinking_mean = drinking.groupby(['Group']).mean()
drinking_mean

# find the difference of the means
mean_diff = drinking_mean.diff().iloc[[1]].values.flatten().tolist()
mean_diff = pd.DataFrame(mean_diff).T
mean_diff.columns = ['age', 'others', 'accident', 'suicide']
mean_diff.rows = 'mean diff'
print(mean_diff.iloc[:, 1:])

drinking.head(10)


rdd_drinking = drinking.assign(threshold = (drinking['age'] >= 21).astype(int))
plt.figure(figsize = (12,8))

# age vs others
ax = plt.subplot(3, 1, 1)
model1 = smf.wls('others ~ age * threshold', rdd_drinking).fit()
plt.title('Regression Discontinuity Design')
plt.axvline(x = 21, color = 'r', label = 'Age 21')
plt.axhline(y = drinking_mean.iloc[0,1], xmin = 0, xmax = 0.5, color = 'c')
plt.axhline(y = drinking_mean.iloc[1,1], xmin = 0.5, xmax = 1, color = 'c')
drinking.plot.scatter(x = 'age', y = 'others', ax = ax)
drinking.assign(predictions = model1.fittedvalues).plot(x = 'age', y = 'predictions', ax = ax, color = "gray", linestyle = '--')
plt.legend()

# age vs accident
ax = plt.subplot(3, 1, 2, sharex = ax)
model2 = smf.wls('accident ~ age * threshold', rdd_drinking).fit()
plt.axvline(x = 21, color = 'r', label = 'Age 21')
plt.axhline(y = drinking_mean.iloc[0,2], xmin = 0, xmax = 0.5, color = 'c')
plt.axhline(y = drinking_mean.iloc[1,2], xmin = 0.5, xmax = 1, color = 'c')
drinking.plot.scatter(x = 'age', y = 'accident', ax = ax)
drinking.assign(predictions = model2.fittedvalues).plot(x = 'age', y = 'predictions', ax = ax, color = "gray", linestyle = '--')
plt.legend()

# age vs suicide
ax = plt.subplot(3, 1, 3, sharex = ax)
model3 = smf.wls('suicide ~ age * threshold', rdd_drinking).fit()
plt.axvline(x = 21, color = 'r', label = 'Age 21')
plt.axhline(y = drinking_mean.iloc[0,3], xmin = 0, xmax = 0.5, color = 'c')
plt.axhline(y = drinking_mean.iloc[1,3], xmin = 0.5, xmax = 1, color = 'c')
drinking.plot.scatter(x = 'age', y = 'suicide', ax = ax)
drinking.assign(predictions = model3.fittedvalues).plot(x = 'age', y = 'predictions', ax = ax, color = "gray", linestyle = '--')
plt.legend()
plt.xlim([20, 22])
