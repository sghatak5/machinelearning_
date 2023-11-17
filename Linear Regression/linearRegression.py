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
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# Load the best-performing model from the saved file
pickle_in = open("studentmodel2.pickle", "rb")
linear = pickle.load(pickle_in)

# Print the model's score on the test set
print (linear.score(x_test, y_test))

# Print the coefficients and intercept of the linear regression model
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Make predictions on the test set
prediction = linear.predict(x_test)

# Print the predicted values alongside the actual values
for i in range(len(prediction)):
    print(prediction[i], x_test[i], y_test[i])

# Scatter plot of 'studytime' vs. 'G2' with 'G2' on the y-axis and 'studytime' on the x-axis
p = "G1"
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
