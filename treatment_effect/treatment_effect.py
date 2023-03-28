import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.formula.api as smf


### Treatment effect
### 1. Using hotel_cancellation.csv, write code to estimate the treatment effects if a ‘different room is assigned’ as the treatment indicator and interpret its effect on the room being ‘canceled’. Use all the other columns as the covariates. Write your observations for the results.


# read the data set
hotel_cancellation = pd.read_csv('hotel_cancellation.csv')
hotel_cancellation.head(10)

# change boolean values into numeric values
hotel_cancellation['different_room_assigned'] = hotel_cancellation['different_room_assigned'].astype(int)
hotel_cancellation['is_canceled'] = hotel_cancellation['is_canceled'].astype(int)
hotel_cancellation.head(10)

# assign response and predictor variables
y = hotel_cancellation['is_canceled']
x = hotel_cancellation[['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 
                        'days_in_waiting_list', 'different_room_assigned']]

# fit a logistic regression model
model = sm.Logit(y, x)
result = model.fit()

print(result.params)


### 2. For hotel_cancellation.csv, now use double logistic regression to measure the effect of ‘different room is assigned’ on the room being ‘canceled’.


# first logit regression
y = hotel_cancellation['is_canceled']
x = hotel_cancellation[['lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 
                        'days_in_waiting_list', 'different_room_assigned']]

model1 = sm.Logit(y, x).fit()
print(model1.params)

# second logit regression
y_hat = np.array(model1.predict()).reshape(len(x), 1)
x_new = pd.DataFrame(np.hstack((x, y_hat)), columns = ['lead_time', 'arrival_date_year', 'arrival_date_week_number', 
                                                       'arrival_date_day_of_month', 'days_in_waiting_list', 'different_room_assigned', 
                                                       'y_hat'])

model2 = sm.Logit(y, x_new).fit()
print(model2.params)


### 3. Use bootstrap to estimate the standard error of the treatment effects measured in (2).


# calculate the len of the data
len(hotel_cancellation)

# define the number of bootstrap resamples
n = 1000

# create a matrix to store the treatment effect estimates
treatment = np.zeros((n, model1.params.shape[0]))
treatment

# use bootstrapping to estimate the standard error of the treatment effects
i = 0
for i in range(n):
    resample_index = np.random.choice(hotel_cancellation.index, len(hotel_cancellation), replace = True)
    y_resample = y.iloc[resample_index]
    x_resample = x.iloc[resample_index]
    model1 = sm.Logit(y_resample, x_resample).fit()
    y_hat = np.array(model1.predict()).reshape(len(x_resample), 1)
    x_new = np.hstack((x_resample, y_hat))
    model2 = sm.Logit(y, x_new).fit()
    treatment[i, :] = model2.params[:-1]

# calculate teh standard error of the treatment effect
np.set_printoptions(suppress=True)
treatment_se = treatment.std(axis = 0).round(6)

print('Standrad errors of the treatment effects:')
print(treatment_se)
