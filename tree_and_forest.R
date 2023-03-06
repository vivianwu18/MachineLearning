library(readr)
library(tree)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(caTools)

### Classification Tree (CART)
transaction <- read_csv("Transaction.csv")
transaction$payment_default <- factor(transaction$payment_default)
transaction <- transaction[, -1]

# understand the distribution of the response variable
table(transaction$payment_default)

# split the data set
set.seed(180)
train_index <- sample(nrow(transaction), nrow(transaction) * 0.8)
train_data <- transaction[train_index, ]
test_data <- transaction[-train_index, ]

# use cross validation to build classification tree model
tree_repeat_cv <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
tree_model_cv <- train(payment_default ~ ., data = train_data, method = "rpart", metric = "Accuracy")

tree_model_cv$finalModel

# plot the tree
rpart.plot(tree_model_cv$finalModel)

# evaluate the performance
tree_pred_cv <- predict(object = tree_model_cv, newdata = test_data[-24])
confusionMatrix(factor(tree_pred_cv), factor(test_data$payment_default))


### Random Forest
set.seed(180)
forest_model <- randomForest(payment_default ~ ., data = train_data, ntree = 500, importance = TRUE)

# plot the tree
plot(forest_model)
importance(forest_model)
varImpPlot(forest_model)

# evaluate the performance
forest_pred <- predict(forest_model, newdata = test_data[-24])
confusionMatrix(factor(forest_pred), factor(test_data$payment_default))
