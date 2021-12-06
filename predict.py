import pickle

model_file='model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model=pickle.load(f_in)

#Picking one customer from the dataset
kastama={'gender': 'female',
 'seniorcitizen': 0,
 'partner': 'no',
 'dependents': 'no',
 'phoneservice': 'yes',
 'multiplelines': 'n0_phone_service',
 'internetservice': 'dsl',
 'onlinesecurity': 'yes',
 'onlinebackup': 'no',
 'deviceprotection': 'yes',
 'techsupport': 'yes',
 'streamingtv': 'yes',
 'streamingmovies': 'no',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'mailed_check',
 'tenure': 10,
 'monthlycharges': 69.25,
 'totalcharges': 690.25}


X=dv.transform([kastama])
y_pred=model.predict_proba(X)[0,1]


print('input',kastama)
print("churn probability",y_pred)
