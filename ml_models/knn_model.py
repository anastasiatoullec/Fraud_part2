import joblib
from sklearn.neighbors import KNeighborsClassifier
import fraud as fr


X_train, y_train, X_test, y_test= fr.preparing_data()

knn = KNeighborsClassifier(leaf_size=1,p= 1,n_neighbors=1)
knn.fit(X_train, y_train)
filename = 'knn_model.sav'
joblib.dump(knn, filename)
