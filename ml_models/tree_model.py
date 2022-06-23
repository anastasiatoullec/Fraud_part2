import joblib
from sklearn.tree import DecisionTreeClassifier
import fraud as fr

X_train, y_train, X_test, y_test= fr.preparing_data()

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
filename = 'tree_model.sav'
joblib.dump(tree, filename)
