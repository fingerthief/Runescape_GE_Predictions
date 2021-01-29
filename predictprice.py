import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sqlite3
import csv
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from numpy import mean
from numpy import std
from numpy import absolute
from numpy import arange
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV

def PredictPriceLasso(dayToPredict):
    data = pd.read_csv('armadyl_final.csv')
    data.head()
    # Let’s select some features to explore more :
    data = data[['day','daily']]
    
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
    # fit model
    model.fit(data[['day']], data[['daily']])
    # summarize chosen configuration
    print('alpha: %f' % model.alpha_)
    
    # evaluate model
    scores = cross_val_score(model, data[['day']], data[['daily']], scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = absolute(scores)
    print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))
    # make a prediction
    estimated_price = model.predict([[dayToPredict]])
    # summarize prediction
    print('Estimated Price: %.3f' % estimated_price)
    return estimated_price[0]

#PredictPriceLasso(190)


#Predicting values:
# Function for predicting future values :
def get_regression_predictions(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    return predicted_values

def PredictPrice(dayToPredict):
    data = pd.read_csv('armadyl_final.csv')
    data.head()
    # Let’s select some features to explore more :
    data = data[['day','average']]
    
    # Cases vs Deaths:
    plt.scatter(data['day'] , data['average'] , color='blue')
    plt.xlabel('Day')
    plt.ylabel('Average Price')
    plt.show() 
    
    # Generating training and testing data from our data:
    # We are using 80% data for training.
    train = data[:(int((len(data)*0.9)))]
    test = data[(int((len(data)*0.2))):]
    
    # Modeling:
    # Using sklearn package to model data :
    regr = linear_model.LinearRegression()
    train_x = np.array(train[['day']])
    train_y = np.array(train[['average']])
    regr.fit(train_x,train_y)
    # The coefficients:
    print ('coefficients : ',regr.coef_) #Slope
    print ('Intercept : ',regr.intercept_) #Intercept
    # Plotting the regression line:
    plt.scatter(train['day'], train['average'], color='blue')
    plt.plot(train_x, regr.coef_*train_x + regr.intercept_, '-r')
    plt.xlabel('day')
    plt.ylabel('Average price')
    plt.show()
    
    # Predicting emission for future car:
    day = dayToPredict
    estimated_price = get_regression_predictions(day,regr.intercept_[0],regr.coef_[0][0])
    print ('Estimated Price :',estimated_price)
    
    # Checking various accuracy:
    from sklearn.metrics import r2_score
    test_x = np.array(test[['day']])
    test_y = np.array(test[['average']])
    test_y_ = regr.predict(test_x)
    print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
    print('Mean sum of squares (MSE): %.2f' % np.mean((test_y_ - test_y) ** 2))
    print('R2-score: %.2f' % r2_score(test_y_ , test_y) )
    return estimated_price

PredictPrice(185)