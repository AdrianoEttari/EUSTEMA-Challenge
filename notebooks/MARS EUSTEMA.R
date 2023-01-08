#### Packages and Data ----
library(caret)
library(earth)
library(tidyverse)
library(readxl)
final_train_s_dummies <- read_excel("C:/Users/sales/OneDrive/Desktop/Progetto Eustema/final_train_s_dummies.xlsx", 
                                    col_types = c("numeric", "numeric", "text", 
                                                  "date", "text", "text", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric", "numeric", 
                                                  "numeric", "numeric"))

X = final_train_s_dummies %>%
  select('Tax Related', 'Number of Lawyers',
         'Number of Legal Parties', 'Value formatted',
         'Unified Contribution formatted', 'Milano', 'Bari', 'Bologna', 'Genova',
         'Palermo', 'Napoli', 'Torino', 'Trento', 'Roma', "L'Aquila", 'Potenza',
         'Perugia', 'Campobasso', 'Firenze', 'Cagliari', 'Venezia', 'Cosenza',
         'Ancona', 'Trieste', 'Aosta','OR-140999', 'OR-145009', 'OR-139999',
         'OR-145999', 'OR-130099', 'OR-101003', 'OR-130121', 'OR-130111',
         'OR-130131', 'OR-101002', 'OR-180002', 'OSA-180002')

y = final_train_s_dummies %>% select('Settlement')

X_to_scale = X %>%
  select('Value formatted',
                'Unified Contribution formatted')

X_not_to_scale = X%>%
  select('Number of Lawyers','Number of Legal Parties','Tax Related','Milano', 'Bari', 'Bologna', 'Genova',
                    'Palermo', 'Napoli', 'Torino', 'Trento', 'Roma', "L'Aquila", 'Potenza',
                    'Perugia', 'Campobasso', 'Firenze', 'Cagliari', 'Venezia', 'Cosenza',
                    'Ancona', 'Trieste', 'Aosta','OR-140999', 'OR-145009', 'OR-139999',
                    'OR-145999', 'OR-130099', 'OR-101003', 'OR-130121', 'OR-130111',
                    'OR-130131', 'OR-101002', 'OR-180002', 'OSA-180002')


X_scaled = scale(X_to_scale)

X_scaled_df = data.frame(X_scaled)
X_scaled_df = cbind(X_scaled_df, X_not_to_scale)

#### Split in training and testing set ----
ids_shuffle = sample(nrow(X_scaled_df), nrow(X_scaled_df))
ids_train = ids_shuffle[1:round(nrow(X_scaled_df)*0.9)-1]
ids_test = ids_shuffle[round(nrow(X_scaled_df)*0.9):nrow(X_scaled_df)]

X_train = X_scaled_df[ids_train,]
X_test = X_scaled_df[ids_test,]
y_train = data.frame(y[ids_train,])
y_test = data.frame(y[ids_test,])


X_train = data.matrix(X_train)
X_test = data.matrix(X_test)
y_train = data.matrix(y_train)
y_test = data.matrix(y_test)

#### MARS ----
parameter_grid = floor(expand.grid(degree=1:4, nprune=seq(5,50,5)))

?train

model = train(x=X_train,
              y=as.integer(y_train),
              method='earth',
              metric='RMSE',
              trControl = trainControl(method='cv', number=10),
              tuneGrid = parameter_grid)

# The best model with respect to RMSE (1818.759) is with nprune=5, degree=2
# The best model with respect to MAE (863.7485) is with nprune=40, degree=3
model$bestTune 

mars_predict = predict(object = model$finalModel,
                       newdata=X_test)

mars_mae = mean(abs(as.integer(y_test)-mars_predict)) # 904.154

#### Let's use the best model with respect to MAE, with BAGGING and see how is the result ----

mars = earth(x=X_train, y=y_train, degree=3, nprune=40)

library(ModelMetrics)
mae(y_train, y_train_pred) # 822.7674
mae(y_test, predict(mars, X_test)) # 847.1232

# Train bagging considering the out-of-bag (merge training and testing set)

Bagging_MARS_oob <- function(X_train, y_train, X_test, y_test,  nbag, oob=T){
  
  if (oob){
    X = rbind(X_train, X_test)
    y = rbind(y_train, y_test)
    rows_X = 1:nrow(X)
    mae_oob = 0
    train_pred_estimators = matrix(0, nrow = nrow(X), ncol = nbag)
    for (i in 1:nbag){
      ids = sample(nrow(X), nrow(X), replace = TRUE)
      ids_oob = rows_X[(rows_X %in% ids)==F] # greatest %in% smallest
      mars = earth(x = X[ids,], y = y[ids], degree = 3, nprune = 40)
      train_pred_estimators[,i] <- predict(mars, newdata = X[ids,])
      mae_oob = mae_oob + mean(abs(y[ids_oob] - predict(mars, newdata = X[ids_oob,])))*1/nbag
    }
    return(list(train_pred = rowMeans(train_pred_estimators), mae_oob = mae_oob))
  } else {
    train_pred_estimators = matrix(0, nrow = nrow(X_train), ncol = nbag)
    test_pred_estimators = matrix(0, nrow = nrow(X_test), ncol = nbag)
    for (i in 1:nbag) {
      ids = sample(nrow(X_train), nrow(X_train), replace = TRUE)
      mars = earth(x = X_train[ids,], y = y_train[ids], degree = 3, nprune = 40)
      train_pred_estimators[,i] <- predict(mars, newdata = X_train[ids,])
      test_pred_estimators[,i] <- predict(mars, newdata = X_test)
    }
    return(list(train_pred = rowMeans(train_pred_estimators), test_pred = rowMeans(test_pred_estimators)))
  }
}

rm(final_train_s_dummies, ids_shuffle, ids_test, ids_train, X, X_not_to_scale, X_scaled, X_scaled_df, 
   X_to_scale, y)

model <- Bagging_MARS_oob(X_train, y_train, X_test, y_test, 50) #mae_oob = 1046.091
mae(y_train, model$train_pred) #1616.249

model2 <- Bagging_MARS_oob(X_train, y_train, X_test, y_test, 50, FALSE)
#Warning messages:
# 1: In cuts[nterm, ipred] == 0 && !is.null(xrange) && xrange[1, ipred] ==  :
#   'length(x) = 86931 > 1' in coercion to 'logical(1)'
# 2: In cuts[nterm, ipred] == 0 && !is.null(xrange) && xrange[1, ipred] ==  :
#   'length(x) = 86931 > 1' in coercion to 'logical(1)'

mae(y_train, model2$train_pred) #1613.184
mae(y_test, model2$test_pred) #846.4701


