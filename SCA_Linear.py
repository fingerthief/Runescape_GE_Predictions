# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # STOCK PRICE PREDICTION IN PYTHON
# This notebook will contain steps required to perform  simple analysis and predictions on stock prices. The dataset to be used can be gotten from .... which contains stock prices from 2009-05-22 to 2018-08-29. I will be taking the timeseries approach to forecast stock prices for the rest of year 2018 and early parts of year 2019.
# The steps include:
# 1. Importing libraries
# 2. Loading the csv file
# 3. Working with dates
# 4. Exploratory data analysis
# 5. Preprocessing
# 6. Model Building
# %% [markdown]
# # Importing Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime
#get_ipython().run_line_magic('matplotlib', 'inline')

# %% [markdown]
# # Loading the GOOGL.csv file

# %%
train_data = pd.read_csv('data_final.csv')


# %%
train_data.head()


# %%
train_data.info()


# %%
train_data.describe()

# %% [markdown]
# Since we are predicting stock prices at the end of each day, our label(the values to be predicted) would be the Adj. Close column in the dataset.
# %% [markdown]
# # Working with date
# %% [markdown]
# From the train_data.info(), we can see that the train_data.Date has an object data type, so we need to convert it into a datetime object

# %%
train_data.set_index('day', inplace=True)
train_data.head()

# %% [markdown]
# Let's add a few more columns to train_data, containing the year, month and weekday name

# %%
# train_data['Year'] = train_data.index.year
# train_data['Month'] = train_data.index.month
# train_data['Weekday Name'] = train_data.index.day_name()


# %%
train_data.head()

# %% [markdown]
# # EXPLORATORY DATA ANALYSIS
# %% [markdown]
# Now let's visualise the Adjusted Close

# %%
plt.figure(figsize=(11, 8)) # resizing the plot
train_data['daily'].plot()
plt.title('Closing Price History') # adding a title
plt.xlabel('day') # x label
plt.ylabel('Daily Price') # y label
plt.show()


# %%
train_data.boxplot(column=['daily'])

# %% [markdown]
# Visualising other columns in the dataset

# %%
# plt.figure(figsize=(16, 8)) # resizing the plot
# cols = ['Open', 'Close', 'Volume', 'High', 'Low']
# axes = train_data[cols].plot(figsize=(11, 9), subplots = True)
# plt.show()


# %%
cols = ['daily']
for i in cols:
    plt.subplots()
    axes = train_data.boxplot(column= [i])
plt.show()

# %% [markdown]
# Now let's visualise the correlation between these features of the dataset

# %%
corr = train_data.corr()


# %%
sns.heatmap(corr, annot=True)

# %% [markdown]
# Wow! from this, it shows that Adj. Close, Open, High, Close and Low have a very high correlation. So we have to drop these values.
# %% [markdown]
# # PREPROCESSING
# %% [markdown]
# # Adding new features to the dataset
# %% [markdown]
# Due to the high correlation, We have to add some features to the dataset. HL_PCT calculates for the high-low percentage for each day and the PCT_change calculatesfor the open-close percentage for each day. 

# %%
# train_data['HL_PCT'] = (train_data['High'] - train_data['Low']) / train_data['Low'] * 100.0 # high-low percentage
# train_data['PCT_change'] = (train_data['Close'] - train_data['Open']) / train_data['Open'] * 100.0 # open-close percentage


# %%
train_data.shape

# %% [markdown]
# # Checking for null values in the dataset

# %%
train_data.isnull().sum()

# %% [markdown]
# # Picking the features we are working with

# %%
df = train_data[['daily']]

# %% [markdown]
# # Picking forecast data
# %% [markdown]
# Since we want to forecast the stock prices for days and months to come, we are going to shift the Adj. Close column to create room for the predictions of the days to come.

# %%
forecast_out = int(math.ceil(0.05 * len(df))) # forcasting out 5% of the entire dataset
print(forecast_out)
df['label'] = df['daily'].shift(-forecast_out)

# %% [markdown]
# The output shows that we shifted the Adj. close 117 days up. to make room for 117 new predictions. the data shifted would be added to a new column called label and that would be our target values.
# %% [markdown]
# # Model building
# %% [markdown]
# # Import libraries

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# %% [markdown]
# # Scaling the data
# %% [markdown]
# Now we scale the data between -1 and 1  in order to put all columns in the dataset in the same range. We will be using StandardScaler function from the preprocessing module of the sklearn library

# %%
scaler = StandardScaler()


# %%
X = np.array(df.drop(['label'], 1))
scaler.fit(X)
X = scaler.transform(X)

# %% [markdown]
# # Picking data to be predicted
# %% [markdown]
# We have successfully scaled the data. Remember that we included a new column called Label to our dataset which contains forecasted out values. Also, we made room for 117 new predictions. So we are going to pick all rows in the dataset excluding the remaining 117 rows as our training data, and use the remaining 117 rows as the data to be predicted. 

# %%
X_Predictions = X[-forecast_out:] # data to be predicted
X = X[:-forecast_out] # data to be trained


# %%
X.shape

# %% [markdown]
# # Getting the target values

# %%
df.dropna(inplace=True)


# %%
y = np.array(df['label'])
y.shape

# %% [markdown]
# we are going to train the model with 80% of X

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Now we are going to try different linear regression models and see which gives the best accuracy.
# %% [markdown]
# # Linear Regression

# %%
lr = LinearRegression()
lr.fit(X_train, y_train)


# %%
lr_confidence = lr.score(X_test, y_test)


# %%
lr_confidence

# %% [markdown]
# # Random Forest

# %%
rf = RandomForestRegressor()
rf.fit(X_train, y_train)


# %%
rf_confidence = rf.score(X_test, y_test)


# %%
rf_confidence

# %% [markdown]
# # Ridge

# %%
rg = Ridge()
rg.fit(X_train, y_train)


# %%
rg_confidence = rg.score(X_test, y_test)


# %%
rg_confidence

# %% [markdown]
# # SVR

# %%
svr = SVR()
svr.fit(X_train, y_train)


# %%
svr_confidence = svr.score(X_test, y_test)


# %%
svr_confidence

# %% [markdown]
# Now that we have calculated the accuracy for 4 different models, let's visualise which models have the best accuracy.

# %%
names = ['Linear Regression', 'Random Forest', 'Ridge', 'SVR']
columns = ['model', 'accuracy']
scores = [lr_confidence, rf_confidence, rg_confidence, svr_confidence]
alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns = columns)
alg_vs_score


# %%
sns.barplot(data = alg_vs_score, x='model', y='accuracy' )
plt.title('Performance of Different Models')
plt.xticks(rotation='vertical')

# %% [markdown]
# The barplot shows that the RandomForestRegressor has the highest accuracy. Therefore, we would be using the model to predict our X_predict data.
# %% [markdown]
# # Adding the predicted data to the dataset

# %%
last_date = df.index[-1] #getting the lastdate in the dataset
# last_unix = 180 #converting it to time in seconds
one_day = 1 #one day equals 86400 seconds
next_unix = last_date + one_day # getting the time in seconds for the next day


# %%
forecast_set = rf.predict(X_Predictions) # predicting forecast data
df['Forecast'] = np.nan


# %%
for i in forecast_set:
    next_date = next_unix
    next_unix += 1
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

# %% [markdown]
# # Visualizing Adj Close and the Forecast data

# %%
plt.figure(figsize=(18, 8))
df['daily'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('day')
plt.ylabel('Price')
plt.show()


# %%
print(df['Forecast'])
df['Forecast'].plot()


# %%



