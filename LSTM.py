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

def LSTM_Train(columnUsed=[1,2], item='52'):

	# fix random seed for reproducibility
	#numpy.random.seed(7)

	# load the dataset
	#dataframe = read_csv('data_'+item+'.csv', usecols=[1,2], engine='python')
	llist = read_csv('data_joined.csv', nrows=1).columns.tolist()
	llist.remove('day')
	dataframe = read_csv('data_joined.csv', usecols=llist, engine='python')
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
	trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], len(llist)))
	testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], len(llist)))

	# create and fit the LSTM network
	batch_size=4
	steps_per_epoch=math.ceil(len(trainX)/batch_size)
	model = Sequential()
	model.add(LSTM(len(llist)*32, input_shape=(look_back, len(llist))))
	model.add(Dense(len(llist)))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(trainX, trainY, epochs=50, batch_size=batch_size, verbose=1, steps_per_epoch=steps_per_epoch)

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
	for i in range(len(llist)):
		testScore = math.sqrt(mean_squared_error(testY[:,i], testPredict[:,i]))
		print('Test Score: %.2f RMSE' % (testScore))

	model.save('_model')

LSTM_Train()

#with open('items.txt', 'r') as items:
#	for item in items:
#		item = str(item).replace('\n','')
#		LSTM_Train([1,2], item)
#		print ('done with item: ' + item)