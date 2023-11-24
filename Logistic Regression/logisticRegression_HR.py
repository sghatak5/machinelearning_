import numpy as np
import pandas as pd
from matplotlib.pyplot import axis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt, style

# Read data from the CSV file into a Pandas DataFrame
df = pd.read_csv("HR_comma_sep.csv.csv", sep=",")

# Create feature matrix X and target vector y
df = df[["satisfaction_level", "average_montly_hours", "last_evaluation",]]
X = np.array(df)
Y = df.left

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.9)

#Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

#Printing model prediction and accuracy
print(model.predict(X_test))
print(model.score(X_test,Y_test))
