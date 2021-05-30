
import numpy as np
import pandas as pd
import json
from utils import load_data, cal_outliers 

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

def save_df_as_json(data: pd.DataFrame, path: 'file path') ->'json_object':
    """
    ====================================================================
    Convert dataframe to json object.
    """
    return data.to_json(path, orient='records', indent=2)

def load_json_data(path: 'file path') -> dict:
    
    """
    ====================================================================
    Load json object.
    """
    with open(path, 'r') as f:
        json_str = json.load(f)
        return json_str

