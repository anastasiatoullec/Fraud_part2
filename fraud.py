from typing import Any
import pandas as pd
import numpy as np
import warnings
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from collections import Counter
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')



def preparing_data():
        df = pd.read_csv(filepath_or_buffer = '../fraud.csv',
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
        return X_train, y_train, X_test, y_test

def evaluate_model(y_test, y_pred):

    # Calcul de accuracy, precision, recall, f1-score, kappa score and balanced accuracy
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    b = metrics.balanced_accuracy_score(y_test,y_pred)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'b': b }

