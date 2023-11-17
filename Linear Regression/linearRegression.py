import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from matplotlib import style
import pickle

# Read data from the CSV file into a Pandas DataFrame
data = pd.read_csv("student-mat.csv", sep=";")

# Select specific columns for analysis
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Define the target variable (what we want to predict)
predict = "G3"

# Create feature matrix X and target vector y
X = np.array(data.drop([predict], axis=1))
print(X)
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print (linear.score(x_test, y_test))

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

prediction = linear.predict(x_test)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])

p = "studytime"
style.use("ggplot")
plt.scatter(data[p], data["G2"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
