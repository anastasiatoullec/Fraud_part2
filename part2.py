#pip3 install fastapi uvicorn
#pip install passlib[bcrypt]
#uvicorn part2:api --reload
#http://127.0.0.1:8000/docs ou http://localhost:8000/docs

#import joblib,
import uvicorn, requests
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel
import fraud as fr

api = FastAPI(
    title="Fraud detection API",
    description="This API interrogates various machine learning models that detect a fraud operation. Here you can access performances of the algorithms on the test sets. A detailed analyses of this model are presented with this line: https://github.com/Anastasia-ctrl782/Projet_Fraud",
    version="1.0.0", openapi_tags=[
        {
        'name': 'Authorization',
        'description': 'In order to use this Api you need to login'
    },
    {
        'name': 'Logistic Regression',
        
    },
    {
        'name': 'Support vector machines',
        
    }
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
    return "Accuracy: {}\n Precision: {}\n Recall: {}\n F1 Score: {}\n Balanced accuracy: {}".format(fr.log_eval['acc'],
    fr.log_eval['prec'], fr.log_eval['rec'], fr.log_eval['f1'], fr.log_eval['b'])

@api.get("/svm", tags=['Support vector machines'])
def perfomances():
    """Returns perfomances of the model support vector machines
    """
    return "Accuracy: {}\n Precision: {}\n Recall: {}\n F1 Score: {}\n Balanced accuracy: {}".format(fr.svm_eval['acc'],
    fr.svm_eval['prec'], fr.svm_eval['rec'], fr.svm_eval['f1'], fr.svm_eval['b'])

@api.get("/knc", tags=['K Nearest Neighbors Classifier'])
def perfomances():
    """Returns perfomances of the model K Nearest Neighbors Classifier
    """
    return "Accuracy: {}\n Precision: {}\n Recall: {}\n F1 Score: {}\n Balanced accuracy: {}".format(fr.knc_eval['acc'],
    fr.knc_eval['prec'], fr.knc_eval['rec'], fr.knc_eval['f1'], fr.knc_eval['b'])

@api.get("/tree", tags=['Decision tree'])
def perfomances():
    """Returns perfomances of the model decision tree
    """
    return "Accuracy: {}\n Precision: {}\n Recall: {}\n F1 Score: {}\n Balanced accuracy: {}".format(fr.tree_eval['acc'],
    fr.tree_eval['prec'], fr.tree_eval['rec'], fr.tree_eval['f1'], fr.tree_eval['b'])

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
    age: int
    source_Ads: int
    source_Direct: int
    source_SEO: int
    browser_Chrome: int
    browser_FireFox: int
    browser_IE: int
    browser_Opera: int
    browser_Safari: int
    sex_F: int
    sex_M: int

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
    fraud.age,
    fraud.source_Ads,
    fraud.source_Direct,
    fraud.source_SEO,
    fraud.browser_Chrome,
    fraud.browser_FireFox,
    fraud.browser_IE,
    fraud.browser_Opera,
    fraud.browser_Safari,
    fraud.sex_F,
    fraud.sex_M
    ]]
    prediction = log.predict(features).tolist()[0]
    return {
        "prediction":prediction
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
    fraud.age,
    fraud.source_Ads,
    fraud.source_Direct,
    fraud.source_SEO,
    fraud.browser_Chrome,
    fraud.browser_FireFox,
    fraud.browser_IE,
    fraud.browser_Opera,
    fraud.browser_Safari,
    fraud.sex_F,
    fraud.sex_M
    ]]
    prediction = fr.svm.predict(features).tolist()[0]
    return {
        "prediction":prediction
    }

@api.post("/predict3",tags=['K Nearest Neighbors Classifier'])
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
    fraud.age,
    fraud.source_Ads,
    fraud.source_Direct,
    fraud.source_SEO,
    fraud.browser_Chrome,
    fraud.browser_FireFox,
    fraud.browser_IE,
    fraud.browser_Opera,
    fraud.browser_Safari,
    fraud.sex_F,
    fraud.sex_M
    ]]
    prediction = fr.knn.predict(features).tolist()[0]
    return {
        "prediction":prediction
    }

@api.post("/predict4",tags=['Decision tree'])
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
    fraud.age,
    fraud.source_Ads,
    fraud.source_Direct,
    fraud.source_SEO,
    fraud.browser_Chrome,
    fraud.browser_FireFox,
    fraud.browser_IE,
    fraud.browser_Opera,
    fraud.browser_Safari,
    fraud.sex_F,
    fraud.sex_M
    ]]
    prediction = fr.tree.predict(features).tolist()[0]
    return {
        "prediction":prediction
    }
