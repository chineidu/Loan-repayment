from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from utils import load_estimator
from etl import clean_test_data, save_df_as_json, load_json_data, get_IDs, get_data



# instantiate app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    estimator = load_estimator()
    clf = estimator['clf']
    # get data
    new_data = get_data()
    # make predictions
    pred = clf.predict(new_data)
    final_pred = "Yes" if pred[0] == 1 else "No"

    return render_template('index.html', prediction_text=f'{(final_pred)}')

@app.route('/predict_API', methods=['POST'])
def predict_API() -> 'json_object':
    """
    ================================================================
    Get the predictions for new data. The data is a pandas DataFrame.
    It returns json object
    """
    # load the estimator
    estimator = load_estimator()
    clf = estimator['clf']
    le_gender = estimator['le_gender']
    le_married = estimator['le_married']
    le_dep = estimator['le_dep']
    le_edu = estimator['le_edu']
    le_self_emp = estimator['le_self_emp']
    le_pr_ar = estimator['le_pr_ar']

    # get data from postman
    req = request.get_json()
    test_data = {}
    # add values to the dictionary
    test_data['Gender'] = [req['Gender']]
    test_data['Married'] = [req['Married']]
    test_data['Dependents'] = [req['Dependents']]
    test_data['Education'] = [req['Education']]
    test_data['Self_Employed'] = [req['Self_Employed']]
    test_data['ApplicantIncome'] = [req['ApplicantIncome']]
    test_data['CoapplicantIncome'] = [req['CoapplicantIncome']]
    test_data['LoanAmount'] = [req['LoanAmount']]
    test_data['Loan_Amount_Term'] = [req['Loan_Amount_Term']]
    test_data['Credit_History'] = [req['Credit_History']]
    test_data['Property_Area'] = [req['Property_Area']]
    # convert to dataframe
    df = pd.DataFrame.from_dict(test_data, orient='columns')
    new_data = df.to_numpy()
    # encode the categorical features
    new_data[:, 0] = le_gender.transform(new_data[:, 0])
    new_data[:, 1] = le_married.transform(new_data[:, 1])
    new_data[:, 2] = le_dep.transform(new_data[:, 2])
    new_data[:, 3] = le_edu.transform(new_data[:, 3])
    new_data[:, 4] = le_self_emp.transform(new_data[:, 4])
    new_data[:, 10] = le_pr_ar.transform(new_data[:, 10])

    # make predictions
    pred = clf.predict(new_data)
    test_data['Prediction'] = pred
    test_data = test_data['Prediction']
    result = ['Yes' if test_data == 1 else 'No']
    return jsonify({'Prediction': result})

@app.route('/predict_test_data')
def predict_test_data() -> 'json_object':
    """
    ==================================================================
    Get the predictions for new data. The data is a pandas DataFrame.
    It returns json object
    """
    # clean the data
    test_data = clean_test_data('./data/test.csv')
    # convert dataframe to numpy array
    X = test_data.to_numpy()
    # load the estimator
    estimator = load_estimator()
    clf = estimator['clf']
    # make predictions
    pred = clf.predict(X)
    test_data['Predictions'] = pred
    # add the IDs
    id =  get_IDs('./data/test.csv')
    final_df = pd.DataFrame()
    final_df = pd.concat([final_df, id], axis='columns')
    final_df['Predictions'] = pred
    path= './data/pred_data.json'   # path
    # save data as json
    json_data = save_df_as_json(final_df, path)
    # load the json file
    json_data = load_json_data(path)
    return jsonify(json_data)

if __name__ == '__main__':
    app.run(debug=True)
