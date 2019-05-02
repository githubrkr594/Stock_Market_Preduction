
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
ds = pd.read_csv('all_stocks_5yr.csv')
df = ds.loc[ds['Name']=='GOOG']
df= df.reset_index()
df=df.drop(['index','high','low','volume'],axis=1)

train = df[:698]
test = df[698:]


training_data=train[['date','close']]
testing_data=test[['date','close']]

testing_data=testing_data.reset_index()
testing_data=testing_data.drop(['index'],axis=1)

#plotting 
training_data.set_index('date')
training_data.plot(figsize=(10,5))
plt.xlabel('index')
plt.ylabel('closing_price')
plt.legend()

#compairing mean & dev by graph
training_data['close'].rolling(5).mean().plot(label='Data Mean')
training_data['close'].rolling(5).std().plot(label='Data Std')
plt.legend()

#decomposing the data a/c to trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(training_data['close'],model='additive',freq=12)
decomp.plot()

#checking the data if sationary & non-stationary
from statsmodels.tsa.stattools import adfuller

def adf_check(df):
    result = adfuller(df,autolag='AIC')
    if result[1] <= 0.05  :
        print("Stationary")
    else:
        print("Not Stationary")

adf_check(training_data['close'])

#Differencing in case of non stationary
training_data['1st Diff'] = train['close'] - train['close'].shift(1)
training_data['1st Diff'].dropna().plot()
training_data =training_data['1st Diff'].dropna()

 #After 1st diffrencing we checking that data has made stationary or not by adf_test
adf_check(training_data)

#Drawing the ACF diagram for determining MA(q)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(training_data,lags=10)# how to choose right lag value
plt.show()

#Drawing the PACF diagram for determining AR(p)
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(training_data,lags=50)
plt.show()


#01----Applying AR Model
from statsmodels.tsa.arima_model import ARMA
# fitting the data into model
AR_model = ARMA(training_data,order=(1, 0))
AR_result = AR_model.fit()
# making prediction of data
#AR_result.summary()
lag = AR_result.k_ar
pred_AR = AR_result.predict(len())
#Visualising the prediction by plotting
AR_result.plot_predict(start=900,end=981)
#Error by this model
error_by_AR = np.sqrt(np.mean((pred_AR-training_data)**2))

#02----Applying MA Model
from statsmodels.tsa.arima_model import ARMA
# fitting the data into model
MA_model = ARMA(training_data, order=(0, 1))
MA_result = MA_model.fit(disp=False)
#making prediction of data
pred_MA = MA_result .predict(start=0,end=981)
#Visualising the prediction by plotting
MA_result.plot_predict(start=950,end=981)
#Error by this model
error_by_MA = np.sqrt(np.mean((pred_MA-training_data)**2))

#03-----Appling ARMA Model
from statsmodels.tsa.arima_model import ARMA
# fitting the data into model
ARMA_model =ARMA(training_data, order=(1, 1)) 
ARMA_result = ARMA_model.fit(disp=False)
#making prediction of data
pred_ARMA = ARMA_result.predict(start=1,end=981)
#Visualising the prediction by plotting
ARMA_result.plot_predict(start=950,end=981)
#Error by this model
error_by_ARMA = np.sqrt(np.mean((pred_ARMA-training_data)**2))


#04----Applying ARIMA Model
from statsmodels.tsa.arima_model import ARIMA
ARIMA_model= ARIMA(training_data,order=(1,1,0))
ARIMA_result = ARIMA_model.fit()
#making prediction of data
pred_ARIMA = ARIMA_result.predict(start=1,end=981)
#Visualising the prediction by plotting
ARIMA_result.plot_predict(start=950,end=981)
#Error by this model
error_by_ARIMA = np.sqrt(np.mean((pred_ARIMA-training_data)**2))

test['forcast'] = ARIMA_result.predict(start=981,end=1258)
test[['close','forcast']].plot()

#05----Applying SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
SARIMA_model = SARIMAX(training_data, order=(1, 1, 0), seasonal_order=(1, 1, 0, 0))
SARIMA_result = SARIMA_model.fit(disp=False)
#making prediction of data
pred_SARIMA = SARIMA_result.predict(start=1,end=981)
pred_SARIMA.plot()
#Visualising the prediction by plotting
#SARIMA_result.plot_predict(start=950,end=981)
#Error by this model
pred_SARIMA.plot()

error_by_SARIMA = sqrt(mean_squared_error(test.Count, pred.SARIMA))



pred_AR = AR_result.predict(start=960,end=981)
pred_MA = MA_result .predict(start=960,end=981)
pred_ARMA = ARMA_result.predict(start=960,end=981)
pred_ARIMA = ARIMA_result.predict(start=960,end=981)
pred_SARIMA = SARIMA_result.predict(start=960,end=981)

plt.plot(pred_AR,label="AR")
plt.plot(pred_MA,label="MA")
plt.plot(pred_ARMA,label="ARMA")
plt.plot(pred_ARIMA,label="ARIMA")
plt.plot(pred_SARIMA,label="SARIMA")
plt.xlabel('Time')
plt.ylabel('Stock')
plt.suptitle("Compairing all the Models", fontsize=20)
plt.legend()
plt.show()


















