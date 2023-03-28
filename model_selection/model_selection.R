library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(glmnet)
library(moments)


heart <- read_csv("heart.csv", show_col_types = FALSE)

### Model building
# check and remove missing values
summary(heart)

# remove empty columns
heart <- subset(heart, select = -c(7, 11, 15))

# replace NA's with mean
heart <- heart %>%
  mutate(height = replace_na(height, mean(height, na.rm = TRUE))) %>%
  mutate(fat_free_wt = replace_na(fat_free_wt, mean(fat_free_wt, na.rm = TRUE))) %>%
  mutate(chest_dim = replace_na(chest_dim, mean(chest_dim, na.rm = TRUE))) %>%
  mutate(hip_dim = replace_na(hip_dim, mean(hip_dim, na.rm = TRUE))) %>%
  mutate(thigh_dim = replace_na(thigh_dim, mean(thigh_dim, na.rm = TRUE))) %>%
  mutate(biceps_dim = replace_na(biceps_dim, mean(biceps_dim, na.rm = TRUE))) 

summary(heart)
skewness(heart)

# split the data set into training and test data sets
set.seed(1)

sample <- sample(c(TRUE, FALSE), nrow(heart), replace = TRUE, prob = c(0.8, 0.2))
train  <- heart[sample, ]
test   <- heart[!sample, ]

# create the model
model1 <- glm(heart_attack ~ ., data = train)
summary(model1)

xactual <- test[1:16]
ypred_model1 <- predict(model1, newdata = xactual)
yactual <- test$heart_attack

# calculate out-of-sample r-squared
rsq_model1 <- 1 - sum((yactual - ypred_model1) ^ 2)/ sum((yactual - mean(yactual)) ^ 2)
rsq_model1


### Cross validation
set.seed(1)
train_control <- trainControl(method = "cv", number = 8)

model2 <- train(heart_attack ~ ., data = train, trControl = train_control, method = "glm")
summary(model2)
print(model2)
print(model2$resample$Rsquared)


### Lasso regression
ytrain <- train$heart_attack
xtrain <- data.matrix(train[, 1:16])

# find the optimal lambda by using k-fold CV
model3 <- cv.glmnet(xtrain, ytrain, nfolds = 8)
lambda_min <- model3$lambda.min
lambda_min

lambda_1se <- model3$lambda.1se
lambda_1se

plot(model3)

# find out the model with min lambda
model_lambda_min <- glmnet(xtrain, ytrain, lambda = lambda_min)
coef(model_lambda_min)

ypred_lambda_min <- predict(model_lambda_min, s = lambda_min, newx = data.matrix(test[, 1:16]))
yactual <- test$heart_attack
rsq_lambda_min <- 1 - sum((yactual - ypred_lambda_min) ^ 2)/ sum((yactual - mean(yactual)) ^ 2)
rsq_lambda_min

# find out the model with 1se lambda
model_lambda_1se <- glmnet(xtrain, ytrain, lambda = lambda_1se)
coef(model_lambda_1se)

ypred_lambda_1se <- predict(model_lambda_1se, s = lambda_1se, newx = data.matrix(test[, 1:16]))
rsq_lambda_1se <- 1 - sum((yactual - ypred_lambda_1se) ^ 2)/ sum((yactual - mean(yactual)) ^ 2)
rsq_lambda_1se