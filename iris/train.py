from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.5, random_state=1)

clf = SVC()
clf.fit(X_train, y_train)

prediction = clf.predict(X_test[:5])
actual = y_test[:5]
print(prediction, actual)

dump(clf, 'model.joblib')