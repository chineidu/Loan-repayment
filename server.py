from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from utils import load_estimator
from etl import clean_test_data, save_df_as_json, load_json_data, get_IDs



# instantiate app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

    # make predictions
    pred = clf.predict(new_data)
    final_pred = "Yes, you will" if pred[0] == 1 else "No, you won't"

    return render_template('index.html', prediction_text=f'{(final_pred)}')

@app.route('/predict_API')
def predict_API() -> 'json_object':
    """
    ================================================================
    Get the predictions for new data. The data is a pandas DataFrame.
    It returns json object
    """
    # clean the data
    test_data = clean_test_data('./data/test.csv')
    # load estimator
    estimator = load_estimator()
    clf = estimator.get('clf')

    # convert dataframe to numpy array
    X = test_data.to_numpy()

    # make predictions
    pred = clf.predict(X)
    test_data['Predictions'] = pred
    test_data = test_data.reset_index()
        
    path= './data/pred_data.json'   # path
    # save data as json
    json_data = save_df_as_json(test_data, path)
    # load the json file
    json_data = load_json_data(path)

    return jsonify(json_data)

@app.route('/predict_API_2')
def predict_API_2() -> 'json_object':
    """
    ================================================================
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
