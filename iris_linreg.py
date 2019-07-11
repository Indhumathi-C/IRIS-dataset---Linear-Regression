# Importing necessary packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Reading CSV file into pandas dataframe
mypath = "C:/Users/indhu/Desktop/data/"
iris_data = pd.read_csv(mypath+"iris.csv")
iris_data.describe()

# Analysis of label variable and check for outliers
sns.boxplot(iris_data[['Sepal.Length']], data=iris_data)

# Analysis of relationship of independent variable with the label and among themselves
sns.pairplot(iris_data)

# Applying dummies to a non-numeric column to make it usable for machine learning as a feature
Species_dummies = pd.get_dummies(iris_data.Species,prefix="Species")
iris_data = pd.concat([iris_data,Species_dummies],axis=1)
iris_data.head()

# Input feature selection. This list can be varied to try out different set of input features
incols = ['Petal.Length', 'Petal.Width', 'Species_versicolor', 'Species_virginica']

# Creating dataset with selected features
outcol = ['Sepal.Length']
X = iris_data[incols]
Y = iris_data[outcol]

# Train Test Split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

# Applying Linear regression
linmod = LinearRegression()
soln = linmod.fit(x_train,y_train)

# Model's score
print("Score: " + soln.score(x_train,y_train))

# Predict sepal length for the test data set using the developed model
pred = soln.predict(x_test)

# Analysis of the developed model using metrics based on the prediction
print("Mean Error: " + np.sqrt(metrics.mean_squared_error(y_test,pred)))
print("Absolute Error: " + metrics.mean_absolute_error(y_test,pred))
print(" R Squared score: " + metrics.r2_score(y_test,pred))

# Residual plot
xx = np.linspace(1,x_test.shape[0],x_test.shape[0])
res = pred-y_test
plt.scatter(xx,res)
