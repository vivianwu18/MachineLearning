import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt


### Question 1
### Using the sales.csv, write code to show effects of interactions, if any, on the linear regression model to predict the total_sales for a new area using given sales from three areas.


sales = pd.read_csv('sales.csv')
sales.head()

# check for missing data
sales.isnull().mean()

# check for missing data
sales.info()

# split the data
x = sales.iloc[:, 1:4] 
y = sales.iloc[:, 4]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25, random_state = 22)

sales_train = pd.DataFrame(np.column_stack((ytrain, xtrain)), columns = ['total_sales', 'area1_sales', 'area2_sales', 'area3_sales'])
sales_train.head()

sales_test = pd.DataFrame(np.column_stack((ytest, xtest)), columns = ['total_sales', 'area1_sales', 'area2_sales', 'area3_sales'])
sales_test.head()

# create the model without interaction
model = smf.glm(formula = 'total_sales ~ area1_sales + area2_sales + area3_sales', data = sales_train)
result = model.fit()

ypred = result.predict(sales_test)

print('Prediction:')
print(ypred.head())

print(result.summary())

# create the model with interaction
model = smf.glm(formula = 'total_sales ~ area1_sales + area2_sales + area3_sales + area1_sales:area2_sales + area2_sales:area3_sales + area1_sales:area3_sales + area1_sales:area2_sales:area3_sales'
                , data = sales_train)
result = model.fit()

ypred = result.predict(sales_test)

print('Prediction:')
print(ypred.head())

print(result.summary())


# create the model with interaction again and remove insignificant variables
model = smf.glm(formula = 'total_sales ~ area1_sales + area2_sales + area3_sales + area1_sales:area3_sales + area1_sales:area2_sales:area3_sales'
                , data = sales_train)
result = model.fit()

ypred = result.predict(sales_test)

print('Prediction:')
print(ypred.head())

print(result.summary())


### Question 2
### Develop a full Logistic Regression Model using customer.csv to predict whether the customer will purchase the product. Also train trimmed logistic regression models (Trimmed over features in the data). Compute the "in-sample R2" (pseudo) for the models you train and compare the models based on this metric.


customer = pd.read_csv('customer.csv')
customer.head()

# check for missing data
customer.isnull().mean()

# check for missing data
customer.info()

# convert categorical data
customer['Gender'] = np.where(customer['Gender'] == 'Male', 1, 0)
customer.head()

# assign the data
xtrain = customer.iloc[:, 1:4]
ytrain = customer['Purchased']

# scale the estimated salary
scaler = StandardScaler()
xtrain_num = scaler.fit_transform(xtrain[['Age', 'EstimatedSalary']])
xtrain_cate = xtrain['Gender'].to_numpy()
xtrain = np.column_stack((xtrain_cate, xtrain_num))

# model 1: y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3 + x1*x2*x3
train_df = pd.DataFrame(np.column_stack((ytrain, xtrain)), columns = ['y', 'x1', 'x2', 'x3'])
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3 + x1*x2*x3', data = train_df).fit()
print(model.summary())

# model 2: y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3', data = train_df).fit()
print(model.summary())

# model 3: y ~ x1 + x2 + x3 + x1*x2 + x1*x3
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x2 + x1*x3', data = train_df).fit()
print(model.summary())

# model 4: y ~ x1 + x2 + x3 + x1*x2 + x2*x3
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x2 + x2*x3', data = train_df).fit()
print(model.summary())

# model 5: y ~ x1 + x2 + x3 + x1*x2 + x1*x3
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x3 + x2*x3', data = train_df).fit()
print(model.summary())

# model 6: y ~ x1 + x2 + x3 + x1*x2
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x2', data = train_df).fit()
print(model.summary())

# model 7: y ~ x1 + x2 + x3 + x1*x3
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x3', data = train_df).fit()
print(model.summary())

# model 8: y ~ x1 + x2 + x3 + x2*x3
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x2*x3', data = train_df).fit()
print(model.summary())

# model 9: y ~ x1 + x2 + x3
model = smf.logit(formula = 'y ~ x1 + x2 + x3', data = train_df).fit()
print(model.summary())

# model 10: y ~ x1 + x2 + x1*x2
model = smf.logit(formula = 'y ~ x1 + x2 + x1*x2', data = train_df).fit()
print(model.summary())

# model 11: y ~ x1 + x3 + x1*x3
model = smf.logit(formula = 'y ~ x1 + x3 + x1*x3', data = train_df).fit()
print(model.summary())

# model 12: y ~ x2 + x3 + x2*x3
model = smf.logit(formula = 'y ~ x2 + x3 + x2*x3', data = train_df).fit()
print(model.summary())


### Question 3
### For the Logistic Regression models trained above, pick the best model wrt to the in-sample R2 and give your interpretation of the model’s coefficients (For example, what effect does a positive or negative coefficient have on the model and so on).


# model 1: y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3 + x1*x2*x3
train_df = pd.DataFrame(np.column_stack((ytrain, xtrain)), columns = ['y', 'x1', 'x2', 'x3'])
model = smf.logit(formula = 'y ~ x1 + x2 + x3 + x1*x2 + x1*x3 + x2*x3 + x1*x2*x3', data = train_df).fit()
print(model.summary())
print(model.params)

# odds
print('odds:')
i = 0
for i in range(0, len(model.params)):
    print(model.params.index[i], ':', round(math.exp(model.params[i]), 4))
    i = i + 1

# calculate the standard deviation
customer = pd.read_csv('customer.csv')
print(np.std(customer[['Age', 'EstimatedSalary']]))


### Question 4
### Is accuracy a good metric to judge the above model? Give reasons and alternatives to support your answer.

customer = pd.read_csv('customer.csv')

customer[customer['Purchased'] == 1].count()
customer[customer['Purchased'] == 0].count()

fig = plt.bar(['Purchased', 'Not purchased'], [143, 257])
plt.show()


### Question 5
### Plot the interactions of the ‘Age’ and ‘Gender’ features with the ‘Purchased’ output. 


customer = pd.read_csv('customer.csv')

customer['Age Group'] = list(pd.cut(customer['Age'], 2, labels = ['Young', 'Old']))

fig = interaction_plot(customer['Age Group'], customer['Gender'], customer['Purchased'], 
                       colors = ['red', 'blue'], markers = ['o', '^'], ms = 7)
plt.show()

