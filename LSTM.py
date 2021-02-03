import csv
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)

def LSTM_Train(columnUsed, item):

	# fix random seed for reproducibility
	#numpy.random.seed(7)

	# load the dataset
	dataframe = read_csv('data_'+item+'.csv', usecols=['daily','average'], engine='python')
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
	look_back = 10
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)

	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 2))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 2))

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(look_back, 2)))
	model.add(Dense(2))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=20, batch_size=16, verbose=0, steps_per_epoch=4)

	# make predictions
	#trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	#trainPredict = scaler.inverse_transform(trainPredict)
	#trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)

	# calculate root mean squared error
	#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	#print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
	testScore1 = math.sqrt(mean_squared_error(testY[:,1], testPredict[:,1]))

	print('Test Score: %.2f RMSE' % (testScore))
	print('Test Score: %.2f RMSE' % (testScore1))

	model.save('model')


with open('items.txt', 'r') as items:
	for item in items:
		item = str(item).replace('\n','')
		LSTM_Train([1,2], item)
		print ('done with item: ' + item)