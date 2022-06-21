#pip3 install fastapi uvicorn
#pip install passlib[bcrypt]
#uvicorn part2:api --reload
#http://127.0.0.1:8000/docs ou http://localhost:8000/docs

import joblib, uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext
from pydantic import BaseModel

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency 
import warnings
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import statsmodels.api
warnings.filterwarnings('ignore')
from sklearn.svm import LinearSVC
#%matplotlib inline

df = pd.read_csv(filepath_or_buffer = 'fraud.csv',
                           sep = ',',
                           header = 0)
#Nettoyage du jeu de données                          
df['signup_day'] = df['signup_time'].apply(lambda time: time.split( )[0])
df['signup_time'] = df['signup_time'].apply(lambda time: time.split( )[1])
df['purchase_day'] = df['purchase_time'].apply(lambda time: time.split( )[0])
df['purchase_time'] = df['purchase_time'].apply(lambda time: time.split( )[1])

df['signup_year'] = df['signup_day'].apply(lambda date: date.split('-')[0])
df['signup_month'] = df['signup_day'].apply(lambda date: date.split('-')[1])
df['signup_day'] = df['signup_day'].apply(lambda date: date.split('-')[2])


df['purchase_year'] = df['purchase_day'].apply(lambda date: date.split('-')[0])
df['purchase_month'] = df['purchase_day'].apply(lambda date: date.split('-')[1])
df['purchase_day'] = df['purchase_day'].apply(lambda date: date.split('-')[2])

df = df[['user_id','signup_time','signup_day', 'signup_month', 'signup_year',
        'purchase_time','purchase_day', 'purchase_month', 'purchase_year','purchase_value',
        'device_id','source','browser','sex', 'age','ip_address','is_fraud']]

'''
You can find statictical analyses in the folder Projet_fraud : https://github.com/Anastasia-ctrl782/Projet_Fraud

'''
#Oversampling avec la normalisation
df_n = df[['user_id','signup_day', 'signup_month', 'signup_year',
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value',
        'source','browser','sex', 'age','is_fraud']]
#print(df_n.head())

#Definition X et y:
X = df_n.drop(['is_fraud'], axis = 1)
y = df_n.is_fraud

# split into 70:30 ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
# Indexation
X_train = X_train.set_index(i for i in range(105778))
X_test = X_test.set_index(i for i in range(45334))
# out of cat data
X_train_out=X_train.drop(['source', 'browser', 'sex'], axis = 1)
X_test_out=X_test.drop(['source', 'browser', 'sex'], axis = 1)
#print(X_train_out.head())
# Normalisation des données
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_out)
X_test_sc = scaler.fit_transform(X_test_out)

X_train_pd = pd.DataFrame(X_train_sc, columns = ['user_id','signup_day','signup_month','signup_year', 'purchase_day','purchase_month',
                                      'purchase_year','purchase_value','age'])
X_test_pd = pd.DataFrame(X_test_sc, columns = ['user_id','signup_day','signup_month','signup_year', 'purchase_day','purchase_month',
                                      'purchase_year','purchase_value','age'])
#print(X_train_pd.head())

#Encoder les variables catégoriels 
X2_train= pd.get_dummies(X_train[['source','browser','sex']])
X2_test= pd.get_dummies(X_test[['source','browser','sex']])
#print(X2_train.head())

#On ajoute nos donées dans le meme tableau
X_train[['user_id','signup_day', 'signup_month', 'signup_year',
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value',
        'age']]=X_train_pd[['user_id','signup_day', 'signup_month', 'signup_year',
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value','age']]
X_train[['source_Ads','source_Direct','source_SEO','browser_Chrome','browser_FireFox','browser_IE','browser_Opera',
  'browser_Safari','sex_F','sex_M']]=X2_train[['source_Ads','source_Direct','source_SEO','browser_Chrome','browser_FireFox','browser_IE','browser_Opera',
  'browser_Safari','sex_F','sex_M']]
X_test[['user_id','signup_day', 'signup_month', 'signup_year',
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value','age']]=X_test_pd[['user_id','signup_day', 'signup_month', 'signup_year',
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value','age']]
X_test[['source_Ads','source_Direct','source_SEO','browser_Chrome','browser_FireFox','browser_IE','browser_Opera',
  'browser_Safari','sex_F','sex_M']]=X2_test[['source_Ads','source_Direct','source_SEO','browser_Chrome','browser_FireFox','browser_IE','browser_Opera',
  'browser_Safari','sex_F','sex_M']]
#print(X_train.head())

#On enleve extra 'source', 'browser', 'sex'
X_train = X_train.drop(['source', 'browser', 'sex'], axis = 1)
X_test = X_test.drop(['source', 'browser', 'sex'], axis = 1)
#print(X_train.head())

#Oversampling
counter = Counter(y_train)
#print('Before', counter)
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
counter = Counter(y_train)
#print('After', counter)

#Modelization
def evaluate_model(y_test, y_pred):

    # Calcul de accuracy, precision, recall, f1-score, kappa score and balanced accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    b = metrics.balanced_accuracy_score(y_test,y_pred)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'b': b }

#Logistic Regression
# Instanciation d'un premier modèle
log = LogisticRegression()
# Entraînement 
log.fit(X_train, y_train)
# On prédit les y à partir de X_test<strong>Oversampling</strong>
pred = log.predict(X_test)
# Score du modèle
log_eval =  evaluate_model(y_test, pred)
# Matrice de confusion
pd.crosstab(y_test, pred,rownames=['Classe réelle'], colnames=['Classe prédite'])
# On affiche les coefficients obtenus
coeff=log.coef_
# On affiche la constante
intercept=log.intercept_
# On calcule les odd ratios
odd_ratios=np.exp(log.coef_)
# On crée un dataframe qui combine à la fois variables, coefficients et odd-ratios
resultats=pd.DataFrame(X_test.columns.tolist (), columns=["Variables"])
resultats['Coefficients']=log.coef_.tolist()[0]
resultats['Odd_Ratios']=np.exp(log.coef_).tolist()[0]
# On choisit d'afficher les variables avec l'odd ratio le plus élevé et le plus faible
resultats.loc[(resultats['Odd_Ratios']==max(resultats['Odd_Ratios']))|(resultats['Odd_Ratios']==min(resultats['Odd_Ratios']))]
#print(resultats)

api = FastAPI(
    title="Fraud detection API",
    description="This API interrogates various maching learning models.You can access performances of the algorithms on the test sets.",
    version="1.0.0")

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

@api.get("/user")
def current_user(username: str = Depends(get_current_user)):
    return "Hello {}".format(username)



class FraudDetection(BaseModel):
    """
    Input features validation for the ML model
    """
    user_id: float
    signup_day: int
    signup_month: int
    signup_year: int
    purchase_day: int
    purchase_month: int
    purchase_year: int
    purchase_value: int
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

@api.post("/predict")
def predict(fraud:FraudDetection):
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