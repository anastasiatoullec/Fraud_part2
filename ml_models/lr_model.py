import joblib
from sklearn.linear_model import LogisticRegression
import fraud as fr


X_train, y_train, X_test, y_test= fr.preparing_data()

lr = LogisticRegression(random_state = 42, C = 5.428675439323859, penalty = 'l2')
lr.fit(X_train, y_train)
filename = 'lr_model.sav'
joblib.dump(lr, filename)
