from flask import Flask, request, jsonify, render_template
import numpy as np
from utils import load_estimator



# instantiate app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    estimator = load_estimator()
    clf = estimator['clf']
    le_gender = estimator['le_gender']
    le_married = estimator['le_married']
    le_dep = estimator['le_dep']
    le_edu = estimator['le_edu']
    le_self_emp = estimator['le_self_emp']
    le_pr_ar = estimator['le_pr_ar']

    new_data = [[request.form.get("gender"), request.form.get("married"), request.form.get("dependents"), 
                request.form.get("education"), request.form.get("self_employed"), request.form.get("applicant_income"), 
                request.form.get("co_applicant_income"), request.form.get("loan_amount"), request.form.get("loan_amout_term"), 
                request.form.get("credit_history"), request.form.get("property_area")]]
    
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
    final_pred = 'Yes' if pred[0] == 1 else 'No'

    return render_template('index.html', prediction_text=f'Loan Status: {(final_pred)}')

if __name__ == '__main__':
    app.run(debug=True)
