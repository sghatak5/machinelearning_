import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt, style
from sklearn.model_selection import train_test_split

# Read data from the CSV file into a Pandas DataFrame
data = pd.read_csv("insurance_data.csv", sep=",")
#print(data.head())

#Scatter plot with the given data set
"""p = "age"
style.use("ggplot")
plt.scatter(data[p], data["bought_insurance"])
plt.xlabel(p)
plt.ylabel("bought_insurance")
plt.show()"""

feature = "age"
variable = "bought_insurance"

X_train, X_test, Y_train, Y_test = train_test_split(data[["age"]],data.bought_insurance, train_size=0.9)

print(X_test)

model = LogisticRegression()
model.fit(X_train, Y_train)

print(model.predict(X_test))
print(model.score(X_test, Y_test))
