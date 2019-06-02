

library(dplyr)
library(glmnet)
rm(list = ls())

#######################################################################
#### DATA PREPARATION
#######################################################################
## Data import - simulated data
raw_data<-read.csv("path/dataset.csv",header = T,stringsAsFactors = F)

## One-hot encoding for the categorical variables
data<-cbind(raw_data[c(1,2,3,5,7,8)],dummy.data.frame(raw_data[c("Gender","Walk_type","Area")], sep = "."))
names(data)[1]<-"Accelerometer_x"


#######################################################################
#### GLMNET MODEL
#######################################################################

# dependant measure for model
yvar1 <- c("steps")

# predictors to be considered
xvar1 <- c(
  # accelerometer_data
  "Accelerometer_x","Accelerometer_y","Accelerometer_z",
  # Customer_demographics
  "Height", "Weight", "Gender.Female", "Gender.Male", "Walk_type.Brisk", "Walk_type.Slow",
  # Area
  "Area.Flat", "Area.Inclined"
)


# grid search to get best alpha
x_var <- data[, xvar1]
y_var <- data[, yvar1]
set.seed(123456789)
alphas <- seq(from = 0, to  = 1, length.out = 11)
res <- matrix(0, nrow = length(alphas), ncol = 2)
for (a in 1:length(alphas)) {
  set.seed(123456789)
  cvmod <- cv.glmnet(as.matrix(x_var), 
                     as.matrix(y_var),
                     type.measure = "mse",
                     nfolds = 10,
                     alpha = alphas[a])
  
  res[a, c(1, 2)] <- c(alphas[a], (min(cvmod$cvm)))
}
res <- data.frame(res)
names(res)[1] <- "alpha"
names(res)[2] <- "min_error"
res <- res[order(res$min_error), ]
bestalpha <- res[1, 1]

set.seed(123456789)
cvfit <-
  cv.glmnet(
    as.matrix(x_var),
    as.matrix(y_var),
    type.measure = "mse",
    nfolds = 10,
    alpha = bestalpha)

cvfit1 <- as.list(coef(cvfit, s = "lambda.min")[, 1])


y_pred_tr <- predict(cvfit, newx = as.matrix(x_var), s = "lambda.min")
t_final_tr <- as.data.frame(cbind(x_var, y_var, y_pred_tr))

colnames(t_final_tr)[colnames(t_final_tr) == "1"] <- "y_pred"


#######################################################################
#### OUTPUT MEASUREMENT
#######################################################################
##### Since it does not make sense that the prediction of the number of steps 
##### is coming out to be in decimals, so we round it to the nearest integer.

t_final_tr$y_pred <- round(t_final_tr$y_pred)

## Calculation of mape
t_final_tr$abs_error <- abs(t_final_tr$y_pred - t_final_tr$y_var)/t_final_tr$y_var
mape<- mean(t_final_tr$abs_error)
mape <- mean(t_final_tr$abs_error[!is.infinite(t_final_tr$abs_error)])
###the mape is 43.3%, if the data is real then we should apply k-fold cross validation 
# to test the model. Keeping one part as test data and rest as training data. We can 
# also try non-linear, tree based or deep learning based models depending on the trend 
# and distribution of the real data.



