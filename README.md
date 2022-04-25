# Supervised Machine Learning - Predicting Credit Risk

In this assignment, I built a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Steps

### Retrieve the data

In the `Generator` folder in `Resources`, there is a [GenerateData.ipynb](/Resources/Generator/GenerateData.ipynb) notebook that will download data from LendingClub and output two CSVs: 
* `2019loans.csv`
* `2020Q1loans.csv`

I have used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

## Preprocessing: Convert categorical data to numeric

I have created a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. I have also created a testing set from the 2020 loans, using `pd.get_dummies()`. 

## Consider the models

I have created and compared the two models on this data: a logistic regression, and a random forests classifier. Before createing, fitting, and scoreing the models, I made a prediction as to which model will perform better. 

## Fit a LogisticRegression model and RandomForestClassifier model

I created a LogisticRegression model, fitted it to the data, and printed the model's score. I'ce done the same for a RandomForestClassifier. 

## Revisit the Preprocessing: Scale the data

I used `StandardScaler` to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, I made another prediction about how scaling will affect the accuracy of the models.

I fitted and scored the LogisticRegression and RandomForestClassifier models on the scaled data.


