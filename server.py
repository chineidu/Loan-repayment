from flask import Flask, render_template, jsonify
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
