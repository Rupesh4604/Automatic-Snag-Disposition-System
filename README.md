# Automatic-Snag-Disposition-System

Problem Statement : To Design and Develop a Machine Learning Model using Machine Learning Algorithms like Multinomial Naive Bayes, Support Vector Machine, for the prediction of Snag Disposition based on SQMS (SNAG & QUERY MANAGEMENT SYSTEM) dataset. The live Snags taken for the validation of model.

This was a model system developed by me and friend @moola Vishwachadra Rajendar as a part of our internship at AURDC ( Aircraft Upgrade, Research and Development Centre) under DQ, Hindustan Aeronautics Limited (HAL), Ojhar, Nasik India. The Model uses Various Machine Learning Algorithms like Naive Bayes, MultinomialBN, Logistic Regression, Support Vector Machines (SVM), using different kernels, Random Forest Classifer and many more to find the best one that fit the model.

Our Results:

model_name              Accuracy Score 
Linear SVC              94.1864 %
Logistic Regression     87.1613 %

MultinomialNB (after hypertuning and overcoming the Class Imbalance Problem)
Training Accuracy       91.2347 %
Testing Accuracy        95.1142 %


SVM - parameters C=10.0, kernel='linear',degree=3 ,gamma=0.01 
Training Accuracy       95.5495 %
Testing Accuracy        98.7641 %
jjd
