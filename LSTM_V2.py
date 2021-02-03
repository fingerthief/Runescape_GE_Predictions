import csv
from operator import concat
import matplotlib
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import bs4 as bs
from tensorflow.python.keras.layers.core import Dropout

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)

def LSTM_Predict(item, plotNum,itemName):
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# load the dataset
	dataframe = read_csv('data_' + item + '.csv', usecols=['daily','average'], engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	# split into train and test sets
	train_size = int(len(dataset) * 0.75)
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	# reshape into X=t and Y=t+1
	look_back = 10
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],2))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1],2))

	# create and fit the LSTM network
	model = Sequential()
	#model.add(LSTM(4, input_shape=(look_back, 2)))

	# model.fit(trainX, trainY, epochs=2000, batch_size=16, verbose=1, steps_per_epoch=4)
	model.add(LSTM(32, batch_input_shape=(2, look_back, 2), stateful=True, return_sequences=True))
	model.add(LSTM(32, batch_input_shape=(2, look_back, 2), stateful=True))
	model.add(Dense(2))
	model.compile(loss='mean_squared_error', optimizer='adam')

	for i in range(600):
		model.fit(trainX, trainY, epochs=1, batch_size=2, verbose=2, shuffle=False, use_multiprocessing=True)
		model.reset_states()
 
	# make predictions
	trainPredict = model.predict(trainX,batch_size=2)
	testPredict = model.predict(testX,batch_size=2)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)


	#calculate root mean squared error
	testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
	testScore1 = math.sqrt(mean_squared_error(testY[:,1], testPredict[:,1]))

	print('Test Score: %.2f RMSE' % (testScore))
	print('Test Score: %.2f RMSE' % (testScore1))

	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	# plot baseline and predictions
	matplotlib.rcParams.update({'axes.formatter.limits': '-1000,1000'})
	matplotlib.rcParams.update({'axes.formatter.useoffset': False})

	plt.figure(plotNum)
	
	plt.plot(scaler.inverse_transform(dataset),label='Base Data')
	plt.plot(trainPredictPlot,label='Train Prediction')
	plt.plot(testPredictPlot,label='Test Prediction')
	plt.title(itemName + ' | LSTM_V2')
	ShowPlots()
	from sklearn.preprocessing import StandardScaler
	data = pandas.read_csv('data_' + item + '.csv')
	data.head()

	df_new = data[['daily','average']]
	days_to_predict = df_new[-180:].values

	scaler = MinMaxScaler(feature_range=(0, 1))
	last_30_days_scaled = scaler.fit_transform(days_to_predict)

	X_test = []
	X_test.append(last_30_days_scaled)
	X_test_np = numpy.array(X_test)
	X_test_np = numpy.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1],2))

	pred_price = model.predict(X_test_np)
	pred_price = scaler.inverse_transform(pred_price)

	print('Est Price: ' + str(pred_price))
	new_Pred = pred_price
	preds = []
	for x in range(10):
		df_new.loc[len(df_new)] = [new_Pred[0][0],new_Pred[0][1]]

		df_new.index = df_new.index + 1  # shifting index
		df_new = df_new.sort_index()
		days_to_predict = df_new[-180:].values

		scaler = MinMaxScaler(feature_range=(0, 1))
		last_30_days_scaled = scaler.fit_transform(days_to_predict)

		X_test = []
		X_test.append(last_30_days_scaled)
		X_test_np = numpy.array(X_test)
		X_test_np = numpy.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1],2))

		pred_price = model.predict(X_test_np)
		pred_price = scaler.inverse_transform(pred_price)
		new_Pred = pred_price
		preds.append(pred_price)

	return preds

import pandas as pd
def GetItemName(item):
	itemDetails = 'http://services.runescape.com/m=itemdb_rs/api/catalogue/detail.json?item='  + item
	df = pd.read_json(itemDetails)
	return df['item']['name']

def Run_LSTM():
	plotCount = 1
	with open('items.txt', 'r') as items:
		for item in items:
			item = item.replace('\n','')
			itemName = GetItemName(item) + ' | Item: ' + item
			futurePreds = LSTM_Predict(item, plotCount, itemName)
	
			df = pd.read_csv('data_' + item + '.csv')
			df.to_csv('data_' + item + '_predictions.csv')
			count = 181
			plotCount += 1

			with open('data_' + item + '_predictions.csv', 'a+', newline='') as write_obj:
				# Create a writer object from csv module
				csv_writer = csv.writer(write_obj)
				index = 0
				for row in futurePreds:
					csv_writer.writerow([str(count - 1),str(row[0][0]),str(row[0][1]),str(count)])
					index += 1
					count += 1
			plt.title(itemName + ' | LSTM_V2')
			dataframe = read_csv('data_' + item + '_predictions.csv', usecols=[2], engine='python')
			dataset = dataframe.values
			dataset = dataset.astype('float32')
			plt.plot(dataset, label='Average Price Predictions')

			dataframe2 = read_csv('data_' + item + '_predictions.csv', usecols=[1], engine='python')
			dataset2 = dataframe2.values
			dataset2 = dataset2.astype('float32')
			plt.plot(dataset2, label='Daily Price Predictions')

def ShowPlots():
	plt.legend()
	plt.show()


