from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy
import requests
import json

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.5, random_state=1)

endpoint = "http://localhost:8080/seldon/default/iris-model/api/v1.0/predictions"
request = { "data": { "ndarray":  numpy.ndarray.tolist(X_test[:5])} }
response = requests.post(endpoint, json=request)
actual = y_test[:5]
predictions = json.loads(response.text)['data']['ndarray']
print(actual)