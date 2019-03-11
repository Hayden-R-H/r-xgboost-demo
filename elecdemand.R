# :@title: Predicting Electricity Demand in Victoria Using Gradient-Boosted Decision Trees
# :@author: Hayden Reece Hohns
# :@date: 11/03/2019
# :@brief: An application of extreme gradient-boosted decision trees to the problem of forecasting
# electricity demand the following day in Victoria.
# Data acquired from the following link:
# 
# https://vincentarelbundock.github.io/Rdatasets/datasets.html

library(dplyr)
library(ggplot2)
library(tidyr)
library(xgboost)

### Read Data
FILEPATH = "Documents/R-Projects/elecdemand/"
FILENAME = "elecdemand.csv"
elecdemand <- read.csv(paste(FILEPATH, FILENAME, sep=""), 
                       header = TRUE, 
                       sep = ",")

### Wrangle Data
elecdemand$Demand1 <- elecdemand$Demand # Duplicate Column
elecdemand$Demand1 <- lead(elecdemand$Demand1)
elecdemand <- drop_na(elecdemand)
elecdemand <- select(elecdemand, -"X")

### Pre-Process Data
numRows <- nrow(elecdemand)
sizeTrain <- 0.7
sizeTest <- 1 - sizeTrain
numTrain <- floor(sizeTrain * numRows)
numTest <- floor(sizeTest * numRows)
trainSet <- elecdemand[1:numTrain, ]
testSet <- elecdemand[-1:-numTrain, ]

dataTrain <- trainSet[, -dim(trainSet)[2]]
dataTest <- testSet[, -dim(testSet)[2]]
labelTrain <- trainSet$Demand1
labelTest <- testSet$Demand1

rmse <- function(prediction, truth) {
    sqrt(mean((prediction - truth) ** 2))
}

### Train GBDT on data and validate
bst <- xgboost(data = as.matrix(dataTrain), 
               label = labelTrain, 
               max.depth = 8, 
               eta = 1, 
               nthread = 2, 
               nrounds = 20, 
               objective = "reg:linear")

pred <- predict(bst, as.matrix(dataTest))

### Validate Model
RMSE <- rmse(pred, labelTest)
print(RMSE)
outputs = data.frame(cbind(pred, labelTest))
colnames(outputs) = c("Prediction", "Ground Truth")
ggplot(data = outputs, aes(x = labelTest, y = pred)) + 
    geom_point(alpha = 0.5) + 
    scale_size_area() + 
    xlab("Ground Truth") + 
    ylab("Prediction") + 
    ggtitle("Predicting Electricity Demand in Victoria with Decision Trees") + 
    theme(plot.title = element_text(hjust = 0.5))
