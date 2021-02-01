import csv
from operator import concat
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

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def LSTM_Predict(columnUsed,columnToPredictName, item, plotNum):
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# load the dataset
	dataframe = read_csv('data_' + item + '.csv', usecols=[columnUsed], engine='python')
	dataset = dataframe.values
	dataset = dataset.astype('float32')

	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)

	# split into train and test sets
	train_size = int(len(dataset) * 0.75)
	test_size = len(dataset) - train_size
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

	# reshape into X=t and Y=t+1
	look_back = 20
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(look_back, 1)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	#model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2)
	model.fit(trainX, trainY, epochs=2000, batch_size=16, verbose=2, steps_per_epoch=8, use_multiprocessing=True)

	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))

	# shift train predictions for plotting
	trainPredictPlot = numpy.empty_like(dataset)
	trainPredictPlot[:, :] = numpy.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

	# shift test predictions for plotting
	testPredictPlot = numpy.empty_like(dataset)
	testPredictPlot[:, :] = numpy.nan
	testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

	# plot baseline and predictions
	plt.figure(plotNum)
	plt.plot(scaler.inverse_transform(dataset),label='Base Data')
	plt.plot(trainPredictPlot,label='Train Prediction')
	plt.plot(testPredictPlot,label='Test Prediction')
	plt.title(item)
	#plt.show()

	from sklearn.preprocessing import StandardScaler
	data = pandas.read_csv('data_' + item + '.csv')
	data.head()

	df_new = data[[columnToPredictName]]
	last_30_days = df_new[-180:].values

	scaler = MinMaxScaler(feature_range=(0, 1))
	last_30_days_scaled = scaler.fit_transform(last_30_days)

	X_test = []
	X_test.append(last_30_days_scaled)
	X_test_np = numpy.array(X_test)
	X_test_np = numpy.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1],1))

	pred_price = model.predict(X_test_np)
	pred_price = scaler.inverse_transform(pred_price)

	print('Est Price: ' + str(pred_price))
	new_Pred = pred_price
	preds = []
	for x in range(10):
		str_pred = str(new_Pred)
		str_pred = str_pred.replace('[[', '')
		str_pred = str_pred.replace(']]', '')
		df_new.loc[len(df_new)] = float(str_pred)
		df_new.index = df_new.index + 1  # shifting index
		df_new = df_new.sort_index()
		last_30_days = df_new[-180:].values

		scaler = MinMaxScaler(feature_range=(0, 1))
		last_30_days_scaled = scaler.fit_transform(last_30_days)

		X_test = []
		X_test.append(last_30_days_scaled)
		X_test_np = numpy.array(X_test)
		X_test_np = numpy.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1],1))

		pred_price = model.predict(X_test_np)
		pred_price = scaler.inverse_transform(pred_price)
		new_Pred = pred_price
		print('Est Price: ' + str(pred_price))
		preds.append(pred_price)
	return preds

import pandas as pd
plotCount = 1
with open('items.txt', 'r') as items:
	for item in items:
		item = item.replace('\n','')
  
		averagePreds = LSTM_Predict(1,'average',item, plotCount)
		dailyPreds = LSTM_Predict(0,'daily', item,plotCount)
  
		df = pd.read_csv('data_' + item + '.csv')
		df.to_csv('data_' + item + '_predictions.csv')
		count = 181
		plotCount += 1

		with open('data_' + item + '_predictions.csv', 'a+', newline='') as write_obj:
			# Create a writer object from csv module
			csv_writer = csv.writer(write_obj)
			index = 0
			for row in averagePreds:
				str_Average_pred = str(row)
				str_Average_pred = str_Average_pred.replace('[[', '')
				str_Average_pred = str_Average_pred.replace(']]', '')

				str_Daily_Pred = str(dailyPreds[index])
				str_Daily_Pred = str_Daily_Pred.replace('[[', '')
				str_Daily_Pred = str_Daily_Pred.replace(']]', '')

				csv_writer.writerow([str(count - 1),str_Daily_Pred,str_Average_pred,str(count)])
				index += 1
				count += 1

		dataframe = read_csv('data_' + item + '_predictions.csv', usecols=[2], engine='python')
		dataset = dataframe.values
		dataset = dataset.astype('float32')
		plt.plot(dataset, label='Average Price Predictions')

		dataframe2 = read_csv('data_' + item + '_predictions.csv', usecols=[1], engine='python')
		dataset2 = dataframe2.values
		dataset2 = dataset2.astype('float32')
		plt.plot(dataset2, label='Daily Price Predictions')
  
plt.legend()
plt.show()


