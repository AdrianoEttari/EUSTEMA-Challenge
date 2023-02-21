* We want to point out that for each test dataset we made two predictions. Indeed, in test_s and test_od there are NA values. Since we were asked to make predictions on ALL the observations of the test sets we decided to make first predictions using data imputation (prediction_settlement.csv); However, the second predictions are done by removing the NA. 
* In the requirements.txt file there is not the tensorflow library even if it is required to run the notebooks. Once you've created your environment you just need to type in your terminal conda install tensorflow’ and you're done.
* In DATA you can find two datasets that will be used in the relative notebooks (notice that we split the notebooks of preprocessing from the ones of the models). If you don't want to run the whole preprocessing, you can directly run the models notebooks using the csv in DATA.
* Some models need hours to run, so, we provide also the weights of the models in the models folder.

