import numpy as np
import pandas as pd
from matplotlib.pyplot import axis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt, style

# Read data from the CSV file into a Pandas DataFrame
df = pd.read_csv("HR_comma_sep.csv", sep=",")

plt.scatter(df.salary,df.left,marker = '+',color='red')
#plt.show()
# Create feature matrix X and target vector y
Y = df.left
data = df[["average_montly_hours","satisfaction_level","last_evaluation"]]
X = np.array(data)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.9)
print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)
#Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

#Printing model prediction and accuracy
print(model.predict(X_test))
print(model.score(X_test,Y_test))

# Add labels and title
plt.xlabel('Salary')
plt.ylabel('left')
plt.title('Crosstab: Salary vs left')

crossTab = pd.crosstab(df.salary,df.left)
crossTab.plot(kind='bar', color=['green', 'red'])
plt.show()


