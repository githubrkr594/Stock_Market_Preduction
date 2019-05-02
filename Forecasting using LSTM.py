#importing Library
import numpy 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
from sklearn.metrics import mean_squared_error
import math

ds = pd.read_csv('all_stocks_5yr.csv')

dataset = ds.loc[ds['Name']=='AMZN']
dataset= dataset.reset_index()

#df=df.drop(['index','high','low','volume','Name'],axis=1)
dataset['date'] = pd.to_datetime(dataset['date'])
dataset=dataset[['date','close']] 
dataset.set_index('date',inplace=True)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# normalize the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
trainScore  =numpy.sqrt(numpy.mean((trainY[0]-trainPredict[:,0])**2))


testScore =  numpy.sqrt(numpy.mean((testY[0]-testPredict[:,0])**2))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting


# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset),label='dataset')
plt.plot(trainPredictPlot,label='train_prediction')
plt.plot(testPredictPlot,label='test_prediction')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(trainY,label='Training_Data')
plt.plot(trainPredict,label='Train_Prediction')
plt.xlabel('index')
plt.ylabel('closing_price')
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(testPredict,label='Test_Prediction')
plt.plot(testY,label='Testing_data')
plt.xlabel('index')
plt.ylabel('closing_price')
plt.legend()
plt.show()
