# Eustema-Challenge

In this project we have two datasets (train_s.csv and train_od.csv) whose observations are italian lawsuits. We have different variables and we must predict the settlement of the lawsuits (that is in train_s.csv) and the duration and the outcome (in train_od.csv). In the `notebooks` folder you can find 5 notebooks: two notebooks about the preprocessing (one for train_s and one for train_od); two notebooks about 
the models (one for train_s and one for train_od) and one R script for the MARS model which is easier to implement in R than in Python. In the `models` folder you can find the weights of the models (we have two weights for the neural networks for regression and classification of train_od and one weight of the random forest for train_s and one for xgboost for train_s). In the `DATA` folder you can find the datasets
that will be used in the notebooks. In the `requirements.txt` file you can find the libraries that we used to run the notebooks. In the `predictions` folder you can find the predictions of the models. In the `report` folder you can find the report of the project. In `eustema EDA.twb` you can find the Tableau dashboard that we made to explore the data.

## Notes 

* We want to point out that for each test dataset we made two predictions. Indeed, in test_s and test_od there are NA values. Since we were asked to make predictions on ALL the observations of the test sets we decided to make first predictions using data imputation (prediction_settlement.csv); However, the second predictions are done by removing the NA. 
* In the requirements.txt file there is not the tensorflow library even if it is required to run the notebooks. Once you've created your environment you just need to type in your terminal conda install tensorflow’ and you're done.
* In DATA you can find two datasets that will be used in the relative notebooks (notice that we split the notebooks of preprocessing from the ones of the models). If you don't want to run the whole preprocessing, you can directly run the models notebooks using the csv in DATA.
* Some models need hours to run, so, we provide also the weights of the models in the models folder.

