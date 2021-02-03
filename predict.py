import csv
import numpy
from pandas import read_csv
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=10):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return numpy.array(dataX), numpy.array(dataY)

def LSTM_Predict():
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# load the dataset
	#dataframe = read_csv('data_'+item+'.csv', usecols=['daily','average'], engine='python')
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

	model = load_model('_model')
	
	# make predictions
	trainPredict = model.predict(trainX)
	testPredict = model.predict(testX)

	# invert predictions
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform(trainY)
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform(testY)

	df_new = dataframe
	last_30_days = df_new[-180:].values

	scaler = MinMaxScaler(feature_range=(0, 1))

	last_30_days_scaled = scaler.fit_transform(last_30_days)

	X_test = []
	X_test.append(last_30_days_scaled)
	X_test_np = numpy.array(X_test)

	X_test_np = numpy.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1],len(llist)))
	
	pred_price = model.predict(X_test_np)
	pred_price = scaler.inverse_transform(pred_price)

	print('Est Price: ' + str(pred_price))
	new_Pred = pred_price
	preds = []
	for x in range(10):
		p = []
		for ii in range(len(new_Pred[0])):
			p.append(new_Pred[0][ii])
		df_new.loc[len(df_new)] = p
		df_new.index = df_new.index + 1 # shifting index
		df_new = df_new.sort_index()
		last_30_days = df_new[-180:].values

		scaler = MinMaxScaler(feature_range=(0, 1))

		last_30_days_scaled = scaler.fit_transform(last_30_days)

		X_test = []
		X_test.append(last_30_days_scaled)
		X_test_np = numpy.array(X_test)

		X_test_np = numpy.reshape(X_test_np, (X_test_np.shape[0], X_test_np.shape[1],len(llist)))

		pred_price = model.predict(X_test_np)
		pred_price = scaler.inverse_transform(pred_price)
		new_Pred = pred_price
		print('Est Price: ' + str(pred_price))
		preds.append(pred_price)
	return preds


predictions = LSTM_Predict()
df = read_csv('data_joined.csv')
df.to_csv('data_predictions.csv', index=None)
count = 181
with open('data_predictions.csv', 'a+', newline='') as write_obj:
	# Create a writer object from csv module
	csv_writer = csv.writer(write_obj)

	# build row
	
	for i in range(len(predictions)):
		str_pred = [str(count - 1)]
		one = predictions
		two = predictions[i]
		three = predictions[i][0]
		for ei in range(len(predictions[i][0])):
			str_pred.append(predictions[i][0][ei])
		
		csv_writer.writerow(str_pred)
		count += 1