# Loan Repayment Prediction

## Project Summary

* In this project, the various factors affecting loan repayment were clearfully analysed. The analysis showed that ***Credit History*** and ***Co-applicant  Income*** are the two major factors that affect loan repaymnet. 
* A classification model with an f1-score of **~85%** was built using **Random Forest**.
* The model was saved as a pickle file which was used to build an API for predictions on unseen data.

## Dataset

The dataset has 614 records with 12 features and a target variable **'Loan_Status'**. It contains categorical and numerical features.

## Data Cleaning

Data cleaning was performed because the dataset contains missing values and outliers. Interquartile range was used to detect and remove outliers.

## Data Exploration

* The features do not follow a Normal distribution. It was handled by detecting and removing the ouliers.
![img1](https://i.postimg.cc/VLrhPf5t/img1-2.jpg)

* The important features that affect loan repayment are shown below.
[![img2.jpg](https://i.postimg.cc/tT8MYwMJ/img2.jpg)](https://postimg.cc/cgMmz9ZG)

## Model Building and Performance

## Model Deployment

