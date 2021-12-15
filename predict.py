import pickle
from flask import Flask
from flask import request
from flask import jsonify
<<<<<<< HEAD

=======
>>>>>>> dd2971964e1fbd15afdc0e86eef32d3a7d4ad975

model_file='model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model=pickle.load(f_in)

app=Flask('churn')

@app.route('/predict',methods=['POST'])
<<<<<<< HEAD
def predict():
    kastama=request.get_json()
    
    
    X=dv.transform([kastama])
    y_pred=model.predict_proba(X)[0,1]
    churn=y_pred>=0.5
    
    result={
        'churn_probability':float(y_pred),
        'churn':bool(churn)
    }
=======

def predict():

    kastama=request.get_json()
    X=dv.transform([kastama])
    y_pred=model.predict_proba(X)[0,1]
    churn=(y_pred>=0.5)


    result={
        'churn_probability':float(y_pred),
        'churn':bool(churn)}
    
>>>>>>> dd2971964e1fbd15afdc0e86eef32d3a7d4ad975
    return jsonify(result)




if __name__=="__main__":
<<<<<<< HEAD

    app.run(debug=True, host="0.0.0.0", port=9696)
    

    

=======
    app.run(debug=True, host='0.0.0.0', port=9696)
>>>>>>> dd2971964e1fbd15afdc0e86eef32d3a7d4ad975
