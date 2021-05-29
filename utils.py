import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle

### Helper Functions
def load_data(path):
    """
    ====================================================================
    Load the csv data.
    """
    df = pd.read_csv(path)
    return df 

def cal_outliers(value: str, df: pd.DataFrame) -> List:
    """
    ====================================================================
    Calculate the range of values that are not outliers.
    """    
    q1 = np.percentile(df[value], 25)  # 1st quartile
    q3 = np.percentile(df[value], 75)  # 3rd quartile
    iqr = q3 - q1
    lower = q1 - (1.5 * iqr)
    upper = q3 + (1.5 * iqr)
    val_range = [lower, upper]
    return val_range

def load_estimator() -> 'estimator':
    """
    ====================================================================
    Load the trained model
    """ 
    # load the model
    with open('./model/estimator.pkl', 'rb') as f:
        loaded_estimators = pickle.load(f)
    return loaded_estimators


def clean_n_train_model() -> 'estimator':
    """
    ====================================================================
    1. Clean the data.
    2. Train the model using the cleaned data
    3. Return the trained model with encoders.
    """  
    # load the data
    data = load_data('./data/training.csv')
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
    for col in ['Credit_History', 'Self_Employed', 'Dependents', 'Gender', 'Married']:
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
    # encode the target
    data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})
    # encode other columns
    for enc, col in zip(encoders, cat_cols):
        data[col] = enc.fit_transform(data[col])

    # split the data
    X = data.drop(columns=['Loan_Status'])
    y = data['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    #instantiate model
    clf = RandomForestClassifier(max_depth=2, n_estimators=50, random_state=123)
    # train
    clf.fit(X_train, y_train)

    estimator = {}
    estimator['clf'] = clf
    estimator['le_gender'] = le_gender
    estimator['le_married'] = le_married
    estimator['le_dep'] = le_dep
    estimator['le_edu'] = le_edu
    estimator['le_self_emp'] = le_self_emp
    estimator['le_pr_ar'] = le_pr_ar

    return estimator


if __name__ == '__main__':
    estimator = clean_n_train_model()

    # save estimator
    with open('./model/estimator.pkl', 'wb') as f:
        pickle.dump(estimator, f)

