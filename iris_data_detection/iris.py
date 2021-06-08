#-------- Import Libraries ------
import pandas as pd
import numpy as np
import pickle

#------ Load data from sklearn dataset ------
from sklearn.datasets import load_iris
iris = load_iris()
from sklearn.linear_model import LogisticRegression
#print(iris.keys())


#------ Load data in DataFrame -------
df = pd.DataFrame(iris.data)
df.head()

#----- Define Features of array ------
df.columns = iris.feature_names
df.head()

#----- Define X and Y -----
x = iris.data
y = iris.target
#print(x.shape)
#print(y.shape)

#-------- Train_test_split data ------
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25,random_state=5)

#------ Train data with Logistic Regression -------
mymodel = LogisticRegression(random_state=2)
mymodel.fit(X_train, Y_train)

#------- Prediction -------
#y_pred = mymodel.predict(X_test) #y_test -> output
#print(X_test.ndim)
#print(y_pred)

#------- Save Model to file -------
pickle.dump(mymodel, open('iris.pkl','wb'))


