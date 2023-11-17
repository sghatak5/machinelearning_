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
y = np.array(data[predict])
# Initialize a linear regression model
best = 0
for _ in range(1000):
    # Split the data into training and testing sets (inside the loop for multiple iterations)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Create and train a linear regression model
    linear = sklearn.linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # Evaluate the model's accuracy on the test set
    acc = linear.score(x_test, y_test)

    # Save the model if it performs better than the previous best
    if acc > best:
        best = acc
        with open("studentmodel2.pickle", "wb") as f:
            pickle.dump(linear, f)
