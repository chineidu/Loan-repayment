
import numpy as np
import pandas as pd
import json
from flask import request
from utils import load_data, cal_outliers, load_estimator

def clean_test_data(file: 'csv_file_path') -> 'cleaned_data':
    """
    ====================================================================
    1. Clean the test data.
    2. Return the cleaned_data.
    """  
    from sklearn.preprocessing import LabelEncoder
    # load the data
    data = load_data(file)
    # drop the Loan_ID
    data = data.drop(columns=['Loan_ID'])
    # split the features into categorical and numerical features
    cat_cols = data.select_dtypes(include='object').columns.to_list()
    num_cols = data.select_dtypes(exclude='object').columns.to_list()

    # impute with the median value
    for col in ['Loan_Amount_Term' , 'LoanAmount', 'Credit_History']:
        mean = data[col].median()
        data[col] = np.where(pd.isna(data[col]), mean, data[col])
    # impute the features with the highest occuring value
    for col in ['Credit_History', 'Self_Employed', 'Dependents', 'Gender', 'Married', 'Property_Area']:
        mode = data[col].mode().values[0]
        data[col] = np.where(pd.isna(data[col]), mode, data[col])
    # split the features into categorical and numerical features
    cat_cols = data.select_dtypes(include='object').columns.to_list()
    num_cols = data.select_dtypes(exclude='object').columns.to_list()

    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
        outliers = cal_outliers(col, data)
        # filter out outliers
        data = data.loc[(data[col] > outliers[0]) & (data[col] < outliers[1])]

    # encoders
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    le_dep = LabelEncoder()
    le_edu = LabelEncoder()
    le_self_emp = LabelEncoder()
    le_pr_ar = LabelEncoder()

    encoders = [le_gender, le_married, le_dep, le_edu, le_self_emp, le_pr_ar]      
    # encode other columns
    for enc, col in zip(encoders, cat_cols):
        data[col] = enc.fit_transform(data[col])
    return data

def get_IDs(file: 'csv_file_path') -> np.array:
    """
    =================================================================================
    1. Obtain the loan IDs
    """  
    from sklearn.preprocessing import LabelEncoder
    # load the data
    data = load_data(file)
    # impute with the median value
    for col in ['Loan_Amount_Term' , 'LoanAmount', 'Credit_History']:
        mean = data[col].median()
        data[col] = np.where(pd.isna(data[col]), mean, data[col])
    # impute the features with the highest occuring value
    for col in ['Credit_History', 'Self_Employed', 'Dependents', 'Gender', 'Married', 'Property_Area']:
        mode = data[col].mode().values[0]
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']:
        outliers = cal_outliers(col, data)
        # filter out outliers
        data = data.loc[(data[col] > outliers[0]) & (data[col] < outliers[1])]

    return data['Loan_ID']


def save_df_as_json(data: pd.DataFrame, path: 'file path') ->'json_object':
    """
    =========================================================================
    Convert dataframe to json object.
    """
    return data.to_json(path, orient='records', indent=2)

def load_json_data(path: 'file path') -> dict:
    
    """
    ==========================================================================
    Load json object.
    """
    with open(path, 'r') as f:
        json_str = json.load(f)
        return json_str

def get_data():
    # load the estimator
    estimator = load_estimator()
    clf = estimator['clf']
    le_gender = estimator['le_gender']
    le_married = estimator['le_married']
    le_dep = estimator['le_dep']
    le_edu = estimator['le_edu']
    le_self_emp = estimator['le_self_emp']
    le_pr_ar = estimator['le_pr_ar']

    # get data from the user through the html form
    new_data = [[request.form.get("gender"), request.form.get("married"), request.form.get("dependents"), 
                request.form.get("education"), request.form.get("self_employed"), request.form.get("applicant_income"), 
                request.form.get("co_applicant_income"), request.form.get("loan_amount"), request.form.get("loan_amout_term"), 
                request.form.get("credit_history"), request.form.get("property_area")]]
    # convert to numpy array
    new_data = np.array(new_data)
    # encode the categorical features
    new_data[:, 0] = le_gender.transform(new_data[:, 0])
    new_data[:, 1] = le_married.transform(new_data[:, 1])
    new_data[:, 2] = le_dep.transform(new_data[:, 2])
    new_data[:, 3] = le_edu.transform(new_data[:, 3])
    new_data[:, 4] = le_self_emp.transform(new_data[:, 4])
    new_data[:, 10] = le_pr_ar.transform(new_data[:, 10])
    return new_data
