

import requests

url='http://localhost:9696/predict'


kastama ={
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "no",
    "dependents": "no",
    "phoneservice": "yes",
    "multiplelines": "n0_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "no",
    "deviceprotection": "yes",
    "techsupport": "yes",
    "streamingtv": "yes",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "mailed_check",
    "tenure": 10,
    "monthlycharges": 90.25,
    "totalcharges": 900.25}


response=requests.post(url, json=kastama).json()
print(response)

if response['churn']==True:
    print('sending promo email to %s'%('xyz-123'))

else:
    print('not sending promo email to %s'%('xyz-123'))
