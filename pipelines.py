import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
from imblearn.over_sampling import SMOTE

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

df_n = df[['user_id','signup_day', 'signup_month', 'signup_year', 
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value',
        'source','browser','sex','age','is_fraud']]
#print(df_n.head())

#Definition X et y:
X = df_n.drop(['is_fraud'], axis = 1)
y = df_n.is_fraud

# split into 70:30 ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean'))
      ,('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant'))
      ,('encoder', OrdinalEncoder())
])

numeric_features = ['user_id','signup_day', 'signup_month', 'signup_year',
        'purchase_day', 'purchase_month', 'purchase_year','purchase_value', 'age']

categorical_features = ['source', 'browser', 'sex']

preprocessor = ColumnTransformer(
   transformers=[
    ('numeric', numeric_transformer, numeric_features)
   ,('categorical', categorical_transformer, categorical_features)
]) 

over = SMOTE()

def evaluate_model(y_test, predictions):

    # Calcul de accuracy, precision, recall, f1-score, kappa score and balanced accuracy
    acc = metrics.accuracy_score(y_test, predictions)
    prec = metrics.precision_score(y_test, predictions)
    rec = metrics.recall_score(y_test, predictions)
    f1 = metrics.f1_score(y_test, predictions)
    b = metrics.balanced_accuracy_score(y_test,predictions)
    
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'b': b }


#---------------------------------------Logistic regression-----------------------------------
model = LogisticRegression()
steps = [('preprocessor', preprocessor),('over', over), ('model', model)]
pipeline = Pipeline(steps=steps)
log_model = pipeline.fit(X_train, y_train)
predictions = log_model.predict(X_test)
log_eval =  evaluate_model(y_test, predictions)
print('Accuracy:', log_eval['acc'])
print('Precision:', log_eval['prec'])
print('Recall:', log_eval['rec'])
print('F1 Score:', log_eval['f1'])
print("Balanced accuracy:",log_eval['b'])

joblib.dump(log_model, './log_model.pkl')
joblib.dump(log_eval, './log_eval.pkl')

#------------------------------------------Support vector machines------------------------------------------
model = LinearSVC(random_state=42)
steps = [('preprocessor', preprocessor),('over', over), ('model', model)]
pipeline = Pipeline(steps=steps)
svm_model = pipeline.fit(X_train, y_train)
predictions = svm_model.predict(X_test)
svm_eval =  evaluate_model(y_test, predictions)
print('Accuracy:', svm_eval['acc'])
print('Precision:', svm_eval['prec'])
print('Recall:', svm_eval['rec'])
print('F1 Score:', svm_eval['f1'])
print("Balanced accuracy:",svm_eval['b'])

joblib.dump(svm_model, './svm_model.pkl')
joblib.dump(svm_eval, './svm_eval.pkl')
#---------------------------------------Decision-Tree-Classifier-----------------------------------
model = DecisionTreeClassifier()
steps = [('preprocessor', preprocessor),('over', over), ('model', model)]
pipeline = Pipeline(steps=steps)
dtc_model = pipeline.fit(X_train, y_train)
predictions = dtc_model.predict(X_test)
dtc_eval =  evaluate_model(y_test, predictions)
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print("Balanced accuracy:",dtc_eval['b'])

joblib.dump(dtc_model, './dtc_model.pkl')
joblib.dump(dtc_eval, './dtc_eval.pkl')
#------------------------------------------KNN------------------------------------------
model = KNeighborsClassifier()
steps = [('preprocessor', preprocessor),('over', over), ('model', model)]
pipeline = Pipeline(steps=steps)
knc_model = pipeline.fit(X_train, y_train)
predictions = knc_model.predict(X_test)
knc_eval =  evaluate_model(y_test, predictions)
print('Accuracy:', knc_eval['acc'])
print('Precision:', knc_eval['prec'])
print('Recall:', knc_eval['rec'])
print('F1 Score:', knc_eval['f1'])
print("Balanced accuracy:",knc_eval['b'])

joblib.dump(knc_model, './knc_model.pkl')
joblib.dump(knc_eval, './knc_eval.pkl')








