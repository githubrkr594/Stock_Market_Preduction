# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 02:03:52 2019

@author: Rupesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as sm
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

ds = pd.read_csv('all_stocks_5yr.csv')

df = ds.loc[ds['Name']=='AMZN']
df= df.reset_index()
#df=df.drop(['index','high','low','volume','Name'],axis=1)
df['date'] = pd.to_datetime(df['date'])
df=df[['date','open','close']] 
df.set_index('date',inplace=True)

train = df[:698]
test = df[698:]
 

df['close'].plot(figsize=(10,5))
plt.xlabel('Date')
plt.ylabel('closing_price')
plt.legend()

#compairing mean & dev by graph

train['close'].rolling(12).mean().plot(label='Data Mean')
train['close'].rolling(12).std().plot(label='Data Std')
plt.legend()

#decomposing the data a/c to trend and seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(train['close'],model='additive',freq=12)
decomp.plot()

#Simple Exponential Smothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
pred = test.copy()
model = SimpleExpSmoothing(np.asarray(train['close']))
fit1=model.fit(smoothing_level=0.9,optimized=False)
pred['SES'] = fit1.forecast(len(test))

error_SES =np.sqrt(np.mean((pred ['SES']-test['close'])**2))

#Holt Winter Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
pred = pred.copy()
model = ExponentialSmoothing(np.asarray(train['close']) ,seasonal_periods=4 ,trend='add', seasonal='add',)
fit2 = model.fit()
pred['Holt_Winter'] = fit2.forecast(len(test))

error_HWES =np.sqrt(np.mean((pred['Holt_Winter']-test['close'])**2))

plt.figure(figsize=(10,6))
plt.plot( train['close'], label='Train_data')
plt.plot(test['close'], label='Test_data')
plt.plot(pred['Holt_Winter'], label='Holt_Winter')
plt.plot(pred ['SES'], label='SES')
plt.xlabel('index')
plt.ylabel('closing_price')
plt.legend(loc='best')
plt.suptitle("Compairing all the Models", fontsize=15)
plt.show()







