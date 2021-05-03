## Importing Important Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Importing datset
data = pd.read_csv("./50_Startups.csv")

## Separating Predictor and Target 
X = data.iloc[:, :-2].values
y = data.iloc[:, -1].values

## Encoding categorical data 
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split  # Here we are splitting 80% data as training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)   # Fitting train data into model

pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
#y_pred = regressor.predict(X_test)
print(model.predict(X_test))
