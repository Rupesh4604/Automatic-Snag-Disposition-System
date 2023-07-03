# Automatic-Snag-Disposition-System

This project aims to design and develop a Machine Learning model using algorithms like Multinomial Naive Bayes and Support Vector Machine (SVM) for the prediction of Snag Disposition based on the SNAG & QUERY MANAGEMENT SYSTEM (SQMS) dataset. The model utilizes live snags for validation and evaluation.

## Problem Statement

The problem statement is to create a predictive model that can accurately classify the disposition of snags based on the given SQMS dataset. The model will utilize various machine learning algorithms such as Multinomial Naive Bayes and SVM with different kernels to determine the best-performing algorithm for this task.

## Project Details

This project was developed by me and my friend @moola Vishwachadra Rajendar during our internship at AURDC (Aircraft Upgrade, Research and Development Centre) under DQ, Hindustan Aeronautics Limited (HAL), Ojhar, Nasik, India. The primary goal was to develop an efficient and accurate model using machine learning techniques.

## Results

The following are the results obtained from the different machine learning algorithms used in this project:

| Model Name              | Accuracy Score |
|-------------------------|----------------|
| Linear SVC              | 94.1864%       |
| Logistic Regression     | 87.1613%       |
| MultinomialNB           |                |
| (after hypertuning and overcoming the Class Imbalance Problem)  |                |
| Training Accuracy       | 91.2347%       |
| Testing Accuracy        | 95.1142%       |
| SVM                     |                |
| (parameters: C=10.0, kernel='linear', degree=3, gamma=0.01)  |                |
| Training Accuracy       | 95.5495%       |
| Testing Accuracy        | 98.7641%       |

These results indicate the performance of each algorithm on the task of Snag Disposition prediction. The SVM model with the specified parameters achieved the highest accuracy among the tested algorithms.

Feel free to explore the code and dataset provided in this repository to understand the implementation details and further enhance the system.

**Note**: This project was developed as a part of our internship and aims to provide a solution for snag disposition prediction. The results and accuracy scores may vary depending on the dataset and implementation details.

