# Importing necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Reading the CSV dataset
mypath = "C:/Users/xyz/Desktop/data/"
iris_data = pd.read_csv(mypath+"iris.csv")

# Analysis of the distribution of label and heck for outliers in the label 
sns.boxplot(iris_data[['Sepal.Length']], data=iris_data)

# Analysis of relationship between the individual features in the dataset
sns.pairplot(iris_data)

# Using dummies to convert non-numeric feature to a machine learning usable numeric feature
Species_dummies = pd.get_dummies(iris_data.Species,prefix="Species")
iris_data = pd.concat([iris_data,Species_dummies],axis=1)

# Selecting input features
incols = ['Petal.Length', 'Petal.Width', 'Species_versicolor', 'Species_virginica'] # Different combinations can be tested by varying this
outcol = ['Sepal.Length']
X = iris_data[incols]
Y = iris_data[outcol]

# Train-Test Split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

# Applying linear regression and checking the solution score
linmod = LinearRegression()
soln = linmod.fit(x_train,y_train)
print("Solution Score: " + soln.score(x_train,y_train))

# Predicting sepal length for test data
pred = soln.predict(x_test)

# Analysing the effectiveness of model through metrics
print("Mean Error: " + np.sqrt(metrics.mean_squared_error(y_test,pred)))
print("Absolute Error: " + metrics.mean_absolute_error(y_test,pred))
print("R Squared score: " + metrics.r2_score(y_test,pred))

# Residual Plot
xx = np.linspace(1,x_test.shape[0],x_test.shape[0])
res = pred-y_test
plt.scatter(xx,res)
