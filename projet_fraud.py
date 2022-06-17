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
#print(log.score(X_test, y_test))
#metrics du modele
log_eval =  evaluate_model(y_test, pred)
#print('Accuracy:', log_eval['acc'])
#print('Precision:', log_eval['prec'])
#print('Recall:', log_eval['rec'])
#print('F1 Score:', log_eval['f1'])
#print("Balanced accuracy:",log_eval['b'])
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

#Decision Tree
# Instanciation des modèles
tree = DecisionTreeClassifier()
# Entraînement 
tree.fit(X_train, y_train)
# Prédiction 
pred = tree.predict(X_test)
# Score du modèle
print(tree.score(X_test, y_test))
#metrics du modele
tree_eval =  evaluate_model(y_test, pred)
#print('Accuracy:', tree_eval['acc'])
#print('Precision:', tree_eval['prec'])
#print('Recall:', tree_eval['rec'])
#print('F1 Score:', tree_eval['f1'])
#print("Balanced accuracy:",tree_eval['b'])
# Matrice de confusion
pd.crosstab(y_test, pred,rownames=['Classe réelle'], colnames=['Classe prédite'])

#Algorithme des kNN : Algorithme des k plus proches voisins
# Instanciation des modèles
knc = KNeighborsClassifier()
# Entraînement 
knc.fit(X_train, y_train)
# Prédiction 
pred = knc.predict(X_test)
# Score du modèle
#print(knc.score(X_test, y_test))
#metrics du modele
knc_eval =  evaluate_model(y_test, pred)
#print('Accuracy:', knc_eval['acc'])
#print('Precision:', knc_eval['prec'])
#print('Recall:', knc_eval['rec'])
#print('F1 Score:', knc_eval['f1'])
#print("Balanced accuracy:",knc_eval['b'])
# Matrice de confusion
pd.crosstab(y_test, pred,rownames=['Classe réelle'], colnames=['Classe prédite'])

#SVM : Séparateur à vastes marges
# Instanciation du modèle de SVC
svm = LinearSVC(random_state=42)
# Entraînement 
svm.fit(X_train, y_train)
# Prédiction 
pred = svm.predict(X_test)
# Score du modèle
#print(svm.score(X_test, y_test))
#metrics du modele
svm_eval =  evaluate_model(y_test, pred)
#print('Accuracy:', svm_eval['acc'])
#print('Precision:', svm_eval['prec'])
#print('Recall:', svm_eval['rec'])
#print('F1 Score:', svm_eval['f1'])
#print("Balanced accuracy:",svm_eval['b'])
# Matrice de confusion
pd.crosstab(y_test, pred,rownames=['Classe réelle'], colnames=['Classe prédite'])

'''
modeles_accuracy = [log_eval['b'], tree_eval['b'], knc_eval['b'], svm_eval['b']]
accuracy = pd.DataFrame({"Accuracy":modeles_accuracy,"Algorithmes":["Logistic regression","DecisionTree","KNC","SVM"]})

g = sns.barplot("Accuracy","Algorithmes",data = accuracy,palette="Set3",orient = "h")
g.set_xlabel("Balanced accuracy")
g = g.set_title("Balanced Accuracy scores")
'''
#----------------------------------------------Optimisation du modèle par ses hyperparamètres----------
#----------------------------------------------KNN-----------------------------------------------------
# liste des paramètres à optimiser.
leaf_size = list(range(1,2))
n_neighbors = list(range(1,2))
p=[1,2]
#convertir vers un dictionnaire
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
#isntanciation de KNN
knn_2 = KNeighborsClassifier()
#Utilisation de GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=10)
#Entraînement du modèle
best_model = clf.fit(X_train,y_train)
print('meilleur leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
print('meilleur p:', best_model.best_estimator_.get_params()['p'])
print('meilleur n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

knn = KNeighborsClassifier(leaf_size=1,p= 1,n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

accuracy_score(y_test, pred)

knc_eval =  evaluate_model(y_test, pred)
#print('Accuracy:', knc_eval['acc'])
#print('Precision:', knc_eval['prec'])
#print('Recall:', knc_eval['rec'])
#print('F1 Score:', knc_eval['f1'])
#print("Balanced accuracy:",knc_eval['b'])

confusion_matrix = pd.crosstab(y_test, pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

#----------------------------------------------Logistic regression------------------------------------
Param = {"C": np.logspace(-4, 4, 50), "penalty": ['l1', 'l2']}
grid_search = GridSearchCV(estimator = LogisticRegression(random_state = 42, solver='lbfgs', max_iter=100), param_grid = Param, scoring = "accuracy", cv = 10, verbose = 1, n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
#print("Meilleur score: {:.2f} %".format(best_accuracy*100))
#print("Meilleur parametre:", best_parameters)
clf = LogisticRegression(random_state = 42, C = 5.428675439323859, penalty = 'l2')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
log_eval =  evaluate_model(y_test, pred)
#print('Accuracy:', log_eval['acc'])
#print('Precision:', log_eval['prec'])
#print('Recall:', log_eval['rec'])
#print('F1 Score:', log_eval['f1'])
#print("Balanced accuracy:",log_eval['b'])
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])