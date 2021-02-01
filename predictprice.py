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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv('testData.csv')
data.head()
# Let’s select some features to explore more :
data = data[['day','price']]

# Cases vs Deaths:
plt.scatter(data['day'] , data['price'] , color='blue')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show() 

input('wait')

def testLasso(day):
    from pandas import read_csv
    from sklearn.linear_model import Lasso

    data = pd.read_csv('data_final.csv')
    data.head()
    # Let’s select some features to explore more :
    data = data[['day','daily']]
    X, Y = data[['day']], data[['daily']]
    # define model
    model = Lasso(alpha=1.0)
    # fit model
    model.fit(X, Y)
    # define new data
    row = [day]
    # make a prediction
    yhat = model.predict([row])
    # summarize prediction
    print('Predicted: %.3f' % yhat)
    return yhat[0]

day = 181
count = 1
f = open("testData.txt", "w")
for x in range(60):
    result = testLasso(day)
    day += 1  
    f.write(str(result) + ',' + str(count) + '\n') 
    count += 1

f.close()
    

input('wait')
def PredictPriceLasso(dayToPredict, alphaA=0.99):
    data = pd.read_csv('data_final.csv')
    data.head()
    # Let’s select some features to explore more :
    data = data[['day','daily']]
    x, y = data[['day']], data[['daily']]
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)
    model=Lasso().fit(x,y)
    
    #model = Lasso(alpha=alphaA)
    Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
        normalize=False, positive=False, precompute=False, random_state=None,
        selection='cyclic', tol=0.0001, warm_start=False)
    
    score = model.score(x, y)
    ypred = model.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    print("Alpha:{0:.2f}, R2:{1:.2f}, MSE:{2:.2f}, RMSE:{3:.2f}"
        .format(model.alpha, score, mse, np.sqrt(mse)))
    return ypred[0]

result = PredictPriceLasso(185)
print(str(result))

#Predicting values:
# Function for predicting future values :
def get_regression_predictions(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    return predicted_values

def PredictPrice(dayToPredict):
    data = pd.read_csv('data_final.csv')
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


def PredictPriceLinearMultiple():
    data = pd.read_csv('data_final.csv')
    data.head()

    X = data[['daily','average']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    Y = data['day']
    
    # with sklearn
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)

    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)

    # prediction with sklearn
    day = '185'
    print ('Predicted Item Price: \n', regr.predict([[day]]))

    # with statsmodels
    X = sm.add_constant(X) # adding a constant
    
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(Y) 
    
    print_model = model.summary()
    print(print_model)


#PredictPriceLinearMultiple()

# price  = PredictPrice(385)

# print('')
# print('LINEAR REGRESSION MODEL')
# print(str(price))
# print('')
# i = 181
# testSet = []
# for x in range(150):    
#     otherprice = PredictPriceLasso(i)
#     print('LASSO MODEL')
#     testSet.append(str(otherprice))
#     i += 1
    
# print(testSet)