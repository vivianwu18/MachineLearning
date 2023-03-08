library(readr)
library(dplyr)
library(ggplot2)
library(leaps)
library(fastDummies)
library(car)


### Exploratory data analysis
autos <- read_csv("autos.csv")

### summary for all variables
### numeric data
summary(autos)

### categorical data
# make
ggplot(aes(x = make), data = autos) + geom_bar() + coord_flip()

# fuel_type
ggplot(aes(x = fuel_type), data = autos) + geom_bar()

# aspiration
ggplot(aes(x = aspiration), data = autos) + geom_bar()

# num_of_doors
ggplot(aes(x = num_of_doors), data = autos) + geom_bar()

# body_style
ggplot(aes(x = body_style), data = autos) + geom_bar()

# drive_wheels
ggplot(aes(x = drive_wheels), data = autos) + geom_bar()

# engine_location
ggplot(aes(x = engine_location), data = autos) + geom_bar()

# engine_type
ggplot(aes(x = engine_type), data = autos) + geom_bar()

# fuel_system
ggplot(aes(x = fuel_system), data = autos) + geom_bar()


### Model building
### create dummy variables
autos_df <- dummy_cols(autos, select_columns = c("make", "fuel_type", "aspiration", "num_of_doors", "body_style", "drive_wheels", "engine_location", "engine_type", "fuel_system"), remove_selected_columns = TRUE, remove_first_dummy = TRUE)

### create the model with all variables
model3 <- lm(price ~ ., data = autos_df)
summary(model3)

### remove the variables which have singularity problem
autos_modified1 <- subset(autos, select = -c(fuel_system, engine_type))
autos_modified1_df <- dummy_cols(autos_modified1, select_columns = c("make", "fuel_type", "aspiration", "num_of_doors", "body_style", "drive_wheels", "engine_location"), remove_selected_columns = TRUE, remove_first_dummy = TRUE)

model3_modified1 <- lm(price ~ ., data = autos_modified1_df)
summary(model3_modified1) 

### check the multicolinearity problem by using VIF function and remove the variables with extremely high VIF
vif(model3_modified1)

autos_modified2 <- subset(autos_modified1, select = -c(compression_ratio, fuel_type))
autos_modified2_df <- dummy_cols(autos_modified2, select_columns = c("make", "aspiration", "num_of_doors", "body_style", "drive_wheels", "engine_location"), remove_selected_columns = TRUE, remove_first_dummy = TRUE)

model3_modified2 <- lm(price ~ ., data = autos_modified2_df)
summary(model3_modified2) 

### alpha = 0.01
pvalues_autos <- summary(model3_modified2)$coefficient[, 4]
j <- 1
alpha2 <- 0.1
count2 <- 0
for (j in 1:length(pvalues_autos)) {
  if (pvalues_autos[j] < alpha2){
    count2 = count2 + 1
    j = j + 1
  }
  j = j + 1  
}
count2


### Benjamini-Hochberg Procedure & False Discovery Rate
pvalues_autos_df <- data.frame(pvalues_autos)
q <- 0.1
n <- nrow(pvalues_autos_df)
pvalues_autos_df <- pvalues_autos_df %>%
  arrange(pvalues_autos) %>%
  mutate(rank = 1:n) %>%
  mutate(BHvalues = q * rank / n) %>%
  mutate(test = ifelse(pvalues_autos <= BHvalues, "TRUE", "FALSE"))
head(pvalues_autos_df, 15)

### count significant values
k <- 1
count3 <- 0
for (k in 1:nrow(pvalues_autos_df)) {
  if (pvalues_autos_df$test[k] == "TRUE") {
    count3 = count3 + 1
    k = k + 1
  }
  k = k + 1  
}
count3

plot(pvalues_autos_df$pvalues_autos, log = "xy", col = ifelse(pvalues_autos_df$test == "TRUE", "red", "grey"), pch = 19, xlab = "rank ordered by p-values", ylab = "p-values", main = paste("FDR =", q))
lines(1:n, q*(1:n) / n)

### fdr
fdr <- function(pvals, q, plotit=FALSE){
  pvals <- pvals[!is.na(pvals)]
  N <- length(pvals)
  
  k <- rank(pvals, ties.method="min")
  alpha <- max(pvals[ pvals <= (q*k/N) ])
  
  if(plotit){
    sig <- factor(pvals <= alpha)
    o <- order(pvals)
    plot(pvals[o], log="xy", col = c("grey60","red")[sig[o]], pch=20, 
         ylab="p-values", xlab="tests ordered by p-value", main = paste('FDR =',q))
    lines(1:N, q*(1:N) / N)
  }
  
  return(alpha)
}

### plot the variables
fdr(pvalues_autos, 0.1, plotit = TRUE)
