
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import pickle 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

C=1.0
n_splits=5
output_file=f"model_C={C}.bin"
# data preparation

df=pd.read_csv("/home/leviscoffie/MLzoomcamp/Telco-Churn-Prediction-MLzoomcamp-/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns.str.replace(' ','_').str.lower()
df.columns=df.columns.str.replace(' ','_').str.lower()

categorical_columns=list(df.dtypes[df.dtypes=='object'].index)

categorical_columns

for c in categorical_columns:
    df[c]= df[c].str.lower().str.replace(' ','_')

df.totalcharges=pd.to_numeric(df.totalcharges, errors="coerce")
df.totalcharges=df.totalcharges.fillna(0)


df.churn=(df.churn=='yes').astype(int)


df_full_train, df_test= train_test_split(df, test_size=0.2, random_state=1)

df_train, df_val= train_test_split(df_full_train, test_size=0.25, random_state=1)

numerical=['tenure','monthlycharges','totalcharges']

categorical= ['gender', 'seniorcitizen', 'partner', 'dependents','phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']

df_train=df_train.reset_index(drop=True)
df_val=df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)


y_train=df_train.churn.values
y_val=df_val.churn.values
y_test=df_test.churn.values

y_full_train= df_full_train.churn.values


# %%
del df_train['churn']
del df_val['churn']
del df_test['churn']


# training

def train(df_train,y_train, C=1.0):
    dicts=df_train[categorical+numerical].to_dict(orient='records')
                                   
    dv=DictVectorizer(sparse=False)
    
    
    X_train=dv.fit_transform(dicts)
    
    model=LogisticRegression(solver='liblinear', max_iter=10000 )
    model.fit(X_train,y_train)
    
    return dv, model


# Predicting
def predict(df,dv,model):
    dicts=df[categorical+numerical].to_dict(orient='records')
    X=dv.transform(dicts)
    y_pred=model.predict_proba(X)[:,1]
    
    return y_pred
#VAlidation
# Now calling K-fold cross validation from sklearn
print(f'doing validation with C={C}')
kf=KFold(n_splits=n_splits, shuffle=True, random_state=1) 


kf.split(df_full_train)

train_idx, val_idx=next(kf.split(df_full_train))

len(train_idx), len(val_idx)
fold=0
for train_idx, val_idx in kf.split(df_full_train):

    scores=[]
    
    df_train=df_full_train.iloc[train_idx]
    df_val=df_full_train.iloc[val_idx]

    y_train=df_train.churn.values
    y_val=df_val.churn.values

    dv, model=train(df_train,y_train, C=1.0)
    y_pred=predict(df_val,dv, model)

    auc=roc_auc_score(y_val, y_pred)

    scores.append(auc)

    print(f'auc on fold{fold} is {auc}')
    fold=fold +1

print('validation results:')

print("C=%s %.3f +- %.3f"%(C, np.mean(scores),np.std(scores))) 




print("Training the final model")
dv, model=train(df_full_train,df_full_train.churn.values, C=1.0)
y_pred=predict(df_test,dv, model)


auc=roc_auc_score(y_test, y_pred)

print(f'auc={auc:.3f}')


# Saving the Model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


print(f'the model is saved to {output_file}')




