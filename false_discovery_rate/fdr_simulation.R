library(readr)
library(dplyr)
library(ggplot2)
library(leaps)
library(fastDummies)
library(car)


### Data simulation
set.seed(3531)
data <- rnorm(10010000)
matrix <- matrix(data, nrow = 10000, ncol = 1001)
colnames(matrix) <- 1:1001
hist(matrix)

y <- matrix[, 1]
num <- 1
for (num in 1:1000){
  assign(paste("x", num, sep = ""), matrix[, num + 1])
  num = num + 1
}


### Model building
Regress y on x’s. Is an intercept needed?  Why?  Why not?
  ```{r}
variables <- paste("x", 1:1000, sep = "")
formula1 <- paste("y", paste(variables, collapse = "+"), sep = "~")
formula2 <- paste(formula1, "-1", sep = "")

### with intercept
model1 <- lm(formula1)
summary(model1)

### without intercept
model2 <- lm(formula2)
summary(model2)

### P-values visualization
pvalues <- summary(model2)$coefficients[, 4]
hist(pvalues, main = "Histogram of the p-values" , xlab = "p-value")


### False discovery
### alpha = 0.01
i <- 1
alpha1 <- 0.01
count1 <- 0
for (i in 1:1000) {
  if (pvalues[i] < alpha1){
    count1 = count1 + 1
    i = i + 1
  }
  i = i + 1  
}
count1


### Benjamini-Hochberg procedure
pvalues_df <- data.frame(pvalues)
q <- 0.1
n <- 1001
pvalues_df <- pvalues_df %>%
  arrange(pvalues) %>%
  mutate(rank = 1:1000) %>%
  mutate(BHvalues = q * rank / n) %>%
  mutate(test = ifelse(pvalues <= BHvalues, "TRUE", "FALSE"))

head(pvalues_df, 15)