#pip3 install fastapi uvicorn
#pip install passlib[bcrypt]
#uvicorn api:api --reload
#http://127.0.0.1:8000/docs ou http://localhost:8000/docs

import uvicorn, requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
import joblib
import pandas as pd


api = FastAPI(
    title="Fraud detection API",
    description="This API interrogates various machine learning models that detect a fraud operation. Here you can access performances of the algorithms on the test sets. A detailed analyses of this model are presented with this line: https://github.com/Anastasia-ctrl782/Projet_Fraud",
    version="1.0.0", openapi_tags=[
        {
        'name': 'Authorization',
        'description': 'In order to use this Api you need to login'
    },
    {   'name': 'Logistic Regression' },
    {   'name': 'Support vector machines'},
    {   'name': 'K Nearest Neighbors Classifier'},
    {   'name': 'Decision tree'},
])

security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users = {

    "Alice": {
        "username": "Alice",
        "hashed_password": pwd_context.hash('wonderland'),
    },

    "Clementine" : {
        "username" :  "Clementine",
        "hashed_password" : pwd_context.hash('mandarine'),
    },

    "Bob" : {
        "username" :  "Bob",
        "hashed_password" : pwd_context.hash('builder'),
    }

}

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if not(users.get(username)) or not(pwd_context.verify(credentials.password, users[username]['hashed_password'])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@api.get("/user", tags=['Authorization'])
def current_user(username: str = Depends(get_current_user)):
    """Returns greetings
    """
    return "Hello {}".format(username)

@api.get("/log", tags=['Logistic Regression'])
def perfomances():
    """Returns perfomances of the model Logistic Regression
    """
    log_eval = joblib.load('./log_eval.pkl')
    return {"Accuracy, %": (log_eval['acc']*100).round(2),
           "Precision, %": (log_eval['prec']*100).round(2),
           "Recall, %": (log_eval['rec']*100).round(2),
           "F1 Score, %": (log_eval['f1']*100).round(2),
           "Balanced accuracy, %": (log_eval['b']*100).round(2)
        }

@api.get("/svm", tags=['Support vector machines'])
def perfomances():
    """Returns perfomances of the model support vector machines
    """
    svm_eval = joblib.load('./svm_eval.pkl')
    return {"Accuracy, %": (svm_eval['acc']*100).round(2),
           "Precision, %": (svm_eval['prec']*100).round(2),
           "Recall, %": (svm_eval['rec']*100).round(2),
           "F1 Score, %": (svm_eval['f1']*100).round(2),
           "Balanced accuracy, %": (svm_eval['b']*100).round(2)
        }

@api.get("/tree", tags=['Decision tree'])
def perfomances():
    """Returns perfomances of the model Decision tree
    """
    tree_eval = joblib.load('./tree_eval.pkl')
    return {"Accuracy, %": (tree_eval['acc']*100).round(2),
           "Precision, %": (tree_eval['prec']*100).round(2),
           "Recall, %": (tree_eval['rec']*100).round(2),
           "F1 Score, %": (tree_eval['f1']*100).round(2),
           "Balanced accuracy, %": (tree_eval['b']*100).round(2)
        }


@api.get("/knc", tags=['K Nearest Neighbors Classifier'])
def perfomances():
    """Returns perfomances of the model K Nearest Neighbors Classifier
    """
    knc_eval = joblib.load('./knc_eval.pkl')
    return {"Accuracy, %": (knc_eval['acc']*100).round(2),
           "Precision, %": (knc_eval['prec']*100).round(2),
           "Recall, %": (knc_eval['rec']*100).round(2),
           "F1 Score, %": (knc_eval['f1']*100).round(2),
           "Balanced accuracy, %": (knc_eval['b']*100).round(2)
        }

class FraudDetection(BaseModel):
    """
    Input features validation for the ML model
    """
    user_id: int
    signup_day: int
    signup_month: int
    signup_year: int
    purchase_day: int
    purchase_month: int
    purchase_year: int
    purchase_value: float
    source: str
    browser: str
    sex: str
    age: int

@api.post("/predict1",tags=['Logistic Regression'])
def predictions1(fraud:FraudDetection):
    """
    :param:input data from the post request
    :return predicted type
    """
    features = [[
    fraud.user_id,
    fraud.signup_day,
    fraud.signup_month,
    fraud.signup_year,
    fraud.purchase_day,
    fraud.purchase_month,
    fraud.purchase_year,
    fraud.purchase_value,
    fraud.source,
    fraud.browser,
    fraud.sex,
    fraud.age
    ]]
    log_model = joblib.load('./log_model.pkl')
    new_case = (pd.DataFrame(features, index = ['0'], columns = ['user_id','signup_day', 'signup_month', 'signup_year', 
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value', 'source','browser','sex','age']))     
    prediction = log_model.predict(new_case)
    return {
        "Predicted transaction(1 - fraud, 0 - not fraud)":prediction.tolist()
    }



@api.post("/predict2",tags=['Support vector machines'])
def predictions2(fraud:FraudDetection):
    """
    :param:input data from the post request
    :return predicted type
    """
    features = [[
    fraud.user_id,
    fraud.signup_day,
    fraud.signup_month,
    fraud.signup_year,
    fraud.purchase_day,
    fraud.purchase_month,
    fraud.purchase_year,
    fraud.purchase_value,
    fraud.source,
    fraud.browser,
    fraud.sex,
    fraud.age
    ]]
    svm_model = joblib.load('./svm_model.pkl')
    new_case = (pd.DataFrame(features, index = ['0'], columns = ['user_id','signup_day', 'signup_month', 'signup_year', 
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value', 'source','browser','sex','age']))     
    prediction = svm_model.predict(new_case)
    return {
        "Predicted transaction(1 - fraud, 0 - not fraud)":prediction.tolist()
    }

@api.post("/predict3",tags=['Decision tree'])
def predictions3(fraud:FraudDetection):
    """
    :param:input data from the post request
    :return predicted type
    """
    features = [[
    fraud.user_id,
    fraud.signup_day,
    fraud.signup_month,
    fraud.signup_year,
    fraud.purchase_day,
    fraud.purchase_month,
    fraud.purchase_year,
    fraud.purchase_value,
    fraud.source,
    fraud.browser,
    fraud.sex,
    fraud.age
    ]]
    tree_model = joblib.load('./tree_model.pkl')
    new_case = (pd.DataFrame(features, index = ['0'], columns = ['user_id','signup_day', 'signup_month', 'signup_year', 
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value', 'source','browser','sex','age']))     
    prediction = tree_model.predict(new_case)
    return {
        "Predicted transaction(1 - fraud, 0 - not fraud)":prediction.tolist()
    }



@api.post("/predict4",tags=['K Nearest Neighbors Classifier'])
def predictions4(fraud:FraudDetection):
    """
    :param:input data from the post request
    :return predicted type
    """
    features = [[
    fraud.user_id,
    fraud.signup_day,
    fraud.signup_month,
    fraud.signup_year,
    fraud.purchase_day,
    fraud.purchase_month,
    fraud.purchase_year,
    fraud.purchase_value,
    fraud.source,
    fraud.browser,
    fraud.sex,
    fraud.age
    ]]
    knc_model = joblib.load('./knc_model.pkl')
    new_case = (pd.DataFrame(features, index = ['0'], columns = ['user_id','signup_day', 'signup_month', 'signup_year', 
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value', 'source','browser','sex','age']))     
    prediction = knc_model.predict(new_case)
    return {
        "Predicted transaction(1 - fraud, 0 - not fraud)":prediction.tolist()
    }




class FraudDetection333(BaseModel):
    """
    Input features validation for the ML model
    """
    user_id: int
    signup_day: int
    signup_month: int
    signup_year: int
    purchase_day: int
    purchase_month: int
    purchase_year: int
    purchase_value: float
    source: str
    browser: str
    sex: str
    age: int

@api.post("/predictions_test",tags=['Logistic'])
def predictions_test(fraud:FraudDetection333):
    """
    :param:input data from the post request
    :return predicted type
    """
    features = [[
    fraud.user_id,
    fraud.signup_day,
    fraud.signup_month,
    fraud.signup_year,
    fraud.purchase_day,
    fraud.purchase_month,
    fraud.purchase_year,
    fraud.purchase_value,
    fraud.source,
    fraud.browser,
    fraud.sex,
    fraud.age
    ]]
    log_model = joblib.load('./log_model.pkl')
    new = (pd.DataFrame(featuress, index = ['0'], columns = ['user_id','signup_day', 'signup_month', 'signup_year', 
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value',
        'source','browser','sex','age']))
    prediction = rf_model.predict(new)
    return {
        "Predicted transaction(1 - fraud, 0 - not fraud)":prediction.tolist()
    }

