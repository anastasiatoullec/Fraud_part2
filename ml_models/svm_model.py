import joblib
from sklearn.svm import LinearSVC
import fraud as fr


X_train, y_train, X_test, y_test= fr.preparing_data()

svm = LinearSVC(random_state=42)
svm.fit(X_train, y_train)
filename = 'svm_model.sav'
joblib.dump(svm, filename)
