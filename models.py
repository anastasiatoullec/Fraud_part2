import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import fraud as fr


X_train, y_train, X_test, y_test= fr.preparing_data()

lr = LogisticRegression(random_state = 42, C = 5.428675439323859, penalty = 'l2')
lr.fit(X_train, y_train)
filename = 'lr_model.sav'
joblib.dump(lr, filename)

knn = KNeighborsClassifier(leaf_size=1,p= 1,n_neighbors=1)
knn.fit(X_train, y_train)
filename = 'knn_model.sav'
joblib.dump(knn, filename)

svm = LinearSVC(random_state=42)
svm.fit(X_train, y_train)
filename = 'svm_model.sav'
joblib.dump(svm, filename)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
filename = 'tree_model.sav'
joblib.dump(tree, filename)
