
# coding: utf-8

# # Data reference 
# 


#  https://drive.google.com/file/d/1pP0Rr83ri0voscgr95-YnVCBv6BYV22w/view




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_model import ARIMA, ARMAResults

import datetime
import time
import sys
import seaborn as sns
import statsmodels
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag

from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import normaltest
from matplotlib.pyplot import acorr
plt.style.use('fivethirtyeight')

get_ipython().run_line_magic('matplotlib', 'inline')





df = pd.read_csv('data_stocks.csv')
df.head()





df.rename(columns={'NASDAQ.AAPL': 'AAPL', 'NASDAQ.ADP': 'ADP', 'NASDAQ.CBOE': 'CBOE', 
                   'NASDAQ.CSCO': 'CSCO', 'NASDAQ.EBAY': 'EBAY'
                  }, inplace=True)

df_AAPL = df[["AAPL"]]
df_ADP = df[["ADP"]]
df_CBOE = df[["CBOE"]]
df_CSCO = df[["CSCO"]]
df_EBAY = df[["EBAY"]]


#   Now, predict one stock price at a time

# # (1) Time series forecasting for AAPL




# show plots in the notebook
df_AAPL.plot(figsize=(10,4))


#   Data shows a clear trend, and possible seasonality also <br>
# Since the dataset is NOT STATIONARY, the first step is to use differencing to make it a stationary time series




from pandas import Series
from matplotlib import pyplot
series = df_AAPL
series.hist()
pyplot.show()


#   The histogram shows that the stock prices were not normally distributed




from pandas import Series
series = df_AAPL
X = series.values
L = len(X)
split = int(L / 3)
last = L-split
X1, X2, X3 = X[0:split], X[split+1: last], X[last+1:]
mean1, mean2, mean3 = X1.mean(), X2.mean(), X3.mean()
var1, var2, var3 = X1.var(), X2.var(), X3.var()
print('mean1=         %f, mean2=        %f, mean3=        %f' % (mean1, mean2, mean3))
print('variance1=     %f, variance2=    %f, variance2=    %f' % (var1, var2, var3))


#   The simple test again shows that means and variances are NOT consistent

#   We may use Augmented Dickey Fuller Test to check for the statistical test of unit root (i.e. stationarity)
#   Null Hypothesis: Series is NOT stationary; Alternate Hypothesis: Series is stationary
#  # So, p-value > 0.05 implies NON-STATIONARITY




from pandas import Series
from statsmodels.tsa.stattools import adfuller

series = df['AAPL'].dropna()
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


#   Here, p-value > 0.05, which means Null Hypothesis cannot be rejected, and thus the series is NOT STATIONARY

#   Let us do a simple log transformation to check if it makes it stationary




from pandas import Series
from numpy import log

series = df_AAPL
X = series.values
X = log(X)
L = len(X)
split = int(L / 3)
last = L-split
X1, X2, X3 = X[0:split], X[split+1: last], X[last+1:]
mean1, mean2, mean3 = X1.mean(), X2.mean(), X3.mean()
var1, var2, var3 = X1.var(), X2.var(), X3.var()
print('mean1=         %f, mean2=        %f, mean3=        %f' % (mean1, mean2, mean3))
print('variance1=     %f, variance2=    %f, variance2=    %f' % (var1, var2, var3))

# Checking the plots to see if log transformation has made it stationary or not

df['logAAPL']= np.log(df['AAPL'])
df['logAAPL'].plot(figsize=(10,4))


#   Numbers and chart above show that the log series is NOT stationary

#   First differencing may be the next trial run to see if this makes the series stationary




#df_AAPL.plot(figsize=(10,4))
df['dif_AAPL'] = df['AAPL'] - df['AAPL'].shift(periods=-1)
df['dif_AAPL'].plot(figsize=(10,4))





from pandas import Series
from statsmodels.tsa.stattools import adfuller

series = df['dif_AAPL'].dropna()
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


#   Here, p-value = 0, which is < 0.05; We may reject the NULL Hypothesis; So, the series with first differencing is stationary!!

#   Now, evaluating if the log first difference is also stationary




df['diflogAAPL'] = df['logAAPL'] - df['logAAPL'].shift(periods=-1)
df['diflogAAPL'].plot(figsize=(10,4))

from pandas import Series
from statsmodels.tsa.stattools import adfuller

series = df['diflogAAPL'].dropna()
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


#   Since p-value = 0, this log-first-difference series is STATIONARY; we can now proceed with ARIMA modeling

#  # ---------------------------------------------------------------

#   Now, lets check for autocorrelations




sm.stats.durbin_watson(df['diflogAAPL'].dropna())


# The value of Durbin-Watson statistic is close to 2 if the errors are uncorrelated. <br>
# In this case, it is 2. <br>
# That means that there is a strong evidence that there is zero autocorrelation.<br>




# Lets plot ACF and PACF plots

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['diflogAAPL'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['diflogAAPL'], lags=40, ax=ax2)


#   ACF graph validates that there is zero autocorrelation - so, p = 0
#   PACF graph shows spike only at zero, so q = 0




df['DATE_F'] = df.DATE.apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))

df['logAAPL']= np.log(df['AAPL'])
df['diflogAAPL'] = df['logAAPL'] - df['logAAPL'].shift(periods=-1) # first difference as earlier identified

df = df.dropna()

# Splitting the data into 2 - for model fit and for test

data = df[['DATE','DATE_F','AAPL','logAAPL','diflogAAPL']][:-100]
test = df[['DATE','DATE_F','AAPL','logAAPL','diflogAAPL']][-100:]


#   Using AIC score to identify the best combination of p, d, q parameters for ARIMA




ararray = (data.logAAPL.dropna().as_matrix())

p=0
q=0
d=1
pdq=[]
aic=[]

for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(ararray, (p,d,q)).fit()
            x = model.aic
            x1 = (p,d,q)
            
            print (x1, x)
            aic.append(x)
            pdq.append(x1)
        except:
            pass
                        
keys = pdq
values = aic
d = dict(zip(keys, values))
minaic=min(d, key=d.get)

for i in range(3):
    p=minaic[0]
    d=minaic[1]
    q=minaic[2]
print ("Best Model is :", (p,d,q))


#   Now, to use the identified p, d, q parameters to fit the ARIMA model




ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()





ARIMIAmod.predict(typ = 'levels')





data.loc[1:, 'predict'] = ARIMIAmod.predict(typ='levels')
data.head()





plt.plot(data['logAAPL'][1:], label='original')
plt.plot(data.predict[1:], label='predict')
plt.legend()





numofsteps = 10
stepahead = ARIMIAmod.forecast(numofsteps)[0]
ferrors = ARIMIAmod.forecast(numofsteps)[2]
ferrors





data['error'] = (data['logAAPL'] - data['predict'])
data['sqrError'] = np.square(data['error'])
data['absError'] = np.abs(data['error'])
data.head()





fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.error, label = "Residual")
plt.title("Residuals of Model", size = 10,)
plt.ylabel("Error", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 2)
plt.plot(data.sqrError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Squared", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 3)
plt.plot(data.absError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Abs", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)





fig, ax = plt.subplots(figsize=(10,4))

plot_acf(data.error, lags = 25, ax = ax)
ax.set_title('ACF Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')

fig, ax = plt.subplots(figsize=(10,4))

plot_acf(data.sqrError, lags = 25, ax = ax)
ax.set_title('ACF Squared Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')





fig, ax = plt.subplots(figsize=(4,4))

plot_pacf(data.error, lags = 25, ax = ax)
ax.set_title('Errors PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')

fig, ax = plt.subplots(figsize=(4,4))

plot_pacf(data.sqrError, lags = 25, ax = ax)
ax.set_title('Errors Squared PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')





plt.plot(test.reset_index().logAAPL)
plt.plot(ARIMIAmod.predict(start=data.shape[0], end=data.shape[0]+100, typ='levels'))





plt.plot(ARIMIAmod.predict(start=data.shape[0], end=data.shape[0]+100, typ='levels'))


#   Now, to apply the ARIMA model on test data




test = test.reset_index()





test_logAAPL = list(test.logAAPL.values)





ararray = list(data.logAAPL.values)
test_predict = []
for i in range(99):
    print(i)
    ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()
    test_predict.append(ARIMIAmod.forecast(1)[0])
    ararray.append(test_logAAPL[i])





plt.plot(test_logAAPL)
plt.plot(np.array(test_predict).ravel())





plt.figure(figsize=(10,4))
plt.plot(data.logAAPL, label='original')
plt.plot(data.logAAPL.rolling(300).mean(), label='rolling_mean')
plt.plot(data.logAAPL.rolling(5000).mean(), label='rolling_mean')
plt.legend()





plt.figure(figsize=(10,4))
plt.plot(data.logAAPL.rolling(300).std())





data.head()







def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

print("MFE = ", mean_forecast_err(data.AAPL, np.exp(data.predict)))
print("MAE = ", mean_absolute_err(data.AAPL, np.exp(data.predict)))


#   Mean Forecast error is practically zero; this shows that ARIMA is a great fit

# # -----------------------------------------------------------------------------

# # Second stock: ADP
#   Will repeat the same steps as done for the first stock




df['DATE_F'] = df.DATE.apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df['logADP']= np.log(df['ADP'])
df['diflogADP'] = df['logADP'] - df['logADP'].shift(periods=-1)
df = df.dropna()





# Splitting the data into 2 - for model fit and for test
data = df[['DATE','DATE_F','ADP','logADP','diflogADP']][:-100]
test = df[['DATE','DATE_F','ADP','logADP','diflogADP']][-100:]





print("Size of data ", data.shape)
print("Size of test ", test.shape)





# Plotting the series and its transformation to check for stationarity

fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.ADP, label = "ADP Price")
plt.title("Level Closing Price", size = 20,)
plt.ylabel("Price in Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data.logADP, label = 'Log of ADP', color = 'r')
plt.title("Log of Closing Price", size = 20,)
plt.ylabel("Price in Log Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot( data.diflogADP, label = '1st Difference of Log of ADP', color = 'g')
plt.title("Differenced Log of Closing Price", size = 20,)
plt.ylabel("Differenced Closing Price", size = 10)
plt.xlabel('Day', size = 10)
plt.legend()





#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(data.diflogADP, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


#   Since p-value = 0, this log-first-difference series is STATIONARY; we can now proceed with ARIMA modeling

# First check for autocorrelation




sm.stats.durbin_watson(df['diflogADP'].dropna())


#   Since DW statistic is close to 2, we can say that there is minimum auto-correlation




# Lets plot ACF and PACF plots

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['diflogADP'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['diflogADP'], lags=40, ax=ax2)





# To systematically find the ARIMA model parameters

ararray = (data.logADP.dropna().as_matrix())

p=0
q=0
d=1
pdq=[]
aic=[]

for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(ararray, (p,d,q)).fit()
            x = model.aic
            x1 = (p,d,q)
            
            print (x1, x)
            aic.append(x)
            pdq.append(x1)
        except:
            pass
                        
keys = pdq
values = aic
d = dict(zip(keys, values))
minaic=min(d, key=d.get)

for i in range(3):
    p=minaic[0]
    d=minaic[1]
    q=minaic[2]
print ("Best Model is :", (p,d,q))


#   Use the identified parameters to fit the ARIMA model




ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()





data.loc[1:, 'predict'] = ARIMIAmod.predict(typ='levels')
data.head()





plt.plot(data['logADP'][1:], label='original')
plt.plot(data.predict[1:], label='predict')
plt.legend()





numofsteps = 10
stepahead = ARIMIAmod.forecast(numofsteps)[0]
ferrors = ARIMIAmod.forecast(numofsteps)[2]
ferrors





data['error'] = (data['logADP'] - data['predict'])
data['sqrError'] = np.square(data['error'])
data['absError'] = np.abs(data['error'])
data.head()





fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.error, label = "Residual")
plt.title("Residuals of Model", size = 10,)
plt.ylabel("Error", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 2)
plt.plot(data.sqrError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Squared", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 3)
plt.plot(data.absError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Abs", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)






fig, ax = plt.subplots(figsize=(10,4))

plot_acf(data.error, lags = 25, ax = ax)
ax.set_title('ACF Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')

plot_acf(data.sqrError, lags = 25, ax = ax)
ax.set_title('ACF Squared Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')





fig, ax = plt.subplots(figsize=(10,4))

plot_pacf(data.error, lags = 25, ax = ax)
ax.set_title('Errors PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')

plot_pacf(data.sqrError, lags = 25, ax = ax)
ax.set_title('Errors Squared PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')


#   Now, to run the ARIMA model on test data




test = test.reset_index()
test_logADP = list(test.logADP.values)

ararray = list(data.logADP.values)
test_predict = []
for i in range(99):
    #print(i)
    ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()
    test_predict.append(ARIMIAmod.forecast(1)[0])
    ararray.append(test_logADP[i])
    
plt.plot(test_logADP)
plt.plot(np.array(test_predict).ravel())





plt.figure(figsize=(10,4))
plt.plot(data.logADP, label='original')
plt.plot(data.logADP.rolling(300).mean(), label='rolling_mean')
plt.plot(data.logADP.rolling(5000).mean(), label='rolling_mean')
plt.legend()





plt.figure(figsize=(10,4))
plt.plot(data.logADP.rolling(300).std())





data.head()







def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

print("MFE = ", mean_forecast_err(data.ADP, np.exp(data.predict)))
print("MAE = ", mean_absolute_err(data.ADP, np.exp(data.predict)))


#   Mean Forecast error is practically zero; this shows that ARIMA is a great fit

#   ----------------------

#   Now, for the third stock in portfolio: CBOE




df['DATE_F'] = df.DATE.apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df['logCBOE']= np.log(df['CBOE'])
df['diflogCBOE'] = df['logCBOE'] - df['logCBOE'].shift(periods=-1)
df = df.dropna()





# Splitting the data into 2 - for model fit and for test
data = df[['DATE','DATE_F','CBOE','logCBOE','diflogCBOE']][:-100]
test = df[['DATE','DATE_F','CBOE','logCBOE','diflogCBOE']][-100:]





print("Size of data ", data.shape)
print("Size of test ", test.shape)





# Plotting the series and its transformation to check for stationarity

fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.CBOE, label = "CBOE Price")
plt.title("Level Closing Price", size = 20,)
plt.ylabel("Price in Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data.logCBOE, label = 'Log of CBOE', color = 'r')
plt.title("Log of Closing Price", size = 20,)
plt.ylabel("Price in Log Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot( data.diflogCBOE, label = '1st Difference of Log of CBOE', color = 'g')
plt.title("Differenced Log of Closing Price", size = 20,)
plt.ylabel("Differenced Closing Price", size = 10)
plt.xlabel('Day', size = 10)
plt.legend()





#Perform Dickey-Fuller test TO TEST FOR STATIONARITY:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(data.diflogCBOE, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


#   Since p-value = 0, this log-first-difference series is STATIONARY; we can now proceed with ARIMA modeling




# Now, check for Autocorrelation
sm.stats.durbin_watson(df['diflogCBOE'].dropna())


#   Since DW statistic is close to 2, we can say that there is minimum auto-correlation




# Lets plot ACF and PACF plots

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['diflogCBOE'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['diflogCBOE'], lags=40, ax=ax2)





# To systematically find the ARIMA model parameters

ararray = (data.logCBOE.dropna().as_matrix())

p=0
q=0
d=1
pdq=[]
aic=[]

for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(ararray, (p,d,q)).fit()
            x = model.aic
            x1 = (p,d,q)
            
            print (x1, x)
            aic.append(x)
            pdq.append(x1)
        except:
            pass
                        
keys = pdq
values = aic
d = dict(zip(keys, values))
minaic=min(d, key=d.get)

for i in range(3):
    p=minaic[0]
    d=minaic[1]
    q=minaic[2]
print ("Best Model is :", (p,d,q))





# Now, fit the ARIMA for the best parameters
ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()





data.loc[1:, 'predict'] = np.exp(ARIMIAmod.predict(typ='levels'))
data.head()





plt.plot(data['CBOE'][1:], label='original')
plt.plot(data.predict[1:], label='predict')
plt.legend()





numofsteps = 10
stepahead = ARIMIAmod.forecast(numofsteps)[0]
ferrors = ARIMIAmod.forecast(numofsteps)[2]
ferrors





data['error'] = (data['CBOE'] - data['predict'])
data['sqrError'] = np.square(data['error'])
data['absError'] = np.abs(data['error'])
data.head()





fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.error, label = "Residual")
plt.title("Residuals of Model", size = 10,)
plt.ylabel("Error", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 2)
plt.plot(data.sqrError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Squared", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 3)
plt.plot(data.absError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Abs", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)





fig, ax = plt.subplots(figsize=(10,4))

plot_acf(data.error, lags = 25, ax = ax)
ax.set_title('ACF Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')

plot_acf(data.sqrError, lags = 25, ax = ax)
ax.set_title('ACF Squared Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')





fig, ax = plt.subplots(figsize=(10,4))

plot_pacf(data.error, lags = 25, ax = ax)
ax.set_title('Errors PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')

plot_pacf(data.sqrError, lags = 25, ax = ax)
ax.set_title('Errors Squared PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')


#   Now, run ARIMA model on Test data




test = test.reset_index()
test_logCBOE = list(test.logCBOE.values)

ararray = list(data.logCBOE.values)
test_predict = []
for i in range(99):
    print(i)
    ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()
    test_predict.append(ARIMIAmod.forecast(1)[0])
    ararray.append(test_logCBOE[i])
    
plt.plot(test_logCBOE)
plt.plot(np.array(test_predict).ravel())





plt.figure(figsize=(10,4))
plt.plot(data.logCBOE, label='original')
plt.plot(data.logCBOE.rolling(300).mean(), label='rolling_mean')
plt.plot(data.logCBOE.rolling(5000).mean(), label='rolling_mean')
plt.legend()





plt.figure(figsize=(10,4))
plt.plot(data.logCBOE.rolling(300).std())





data.head()







def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

print("MFE = ", mean_forecast_err(data.CBOE, data.predict))
print("MAE = ", mean_absolute_err(data.CBOE, data.predict))


#   Mean Forecast error is practically zero; this shows that ARIMA is a great fit

# # --------------------------------------------

#   The fourth stock in the list: CSCO




df['DATE_F'] = df.DATE.apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df['logCSCO']= np.log(df['CSCO'])
df['diflogCSCO'] = df['logCSCO'] - df['logCSCO'].shift(periods=-1)
df = df.dropna()




# Splitting the data into 2 - for model fit and for test
data = df[['DATE','DATE_F','CSCO','logCSCO','diflogCSCO']][:-100]
test = df[['DATE','DATE_F','CSCO','logCSCO','diflogCSCO']][-100:]





print("Size of data ", data.shape)
print("Size of test ", test.shape)





# Plotting the series and its transformation to check for stationarity

fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.CSCO, label = "CSCO Price")
plt.title("Level Closing Price", size = 20,)
plt.ylabel("Price in Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data.logCSCO, label = 'Log of CSCO', color = 'r')
plt.title("Log of Closing Price", size = 20,)
plt.ylabel("Price in Log Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot( data.diflogCSCO, label = '1st Difference of Log of CSCO', color = 'g')
plt.title("Differenced Log of Closing Price", size = 20,)
plt.ylabel("Differenced Closing Price", size = 10)
plt.xlabel('Day', size = 10)
plt.legend()





#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(data.diflogCSCO, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


#   Since p-value = 0, this log-first-difference series is STATIONARY; we can now proceed with ARIMA modeling




# Check for autocorrelation
sm.stats.durbin_watson(df['diflogCSCO'].dropna())


#   Since DW statistic is close to 2, we can say that there is minimum auto-correlation




# Lets plot ACF and PACF plots

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['diflogCSCO'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['diflogCSCO'], lags=40, ax=ax2)





# To systematically find the ARIMA model parameters

ararray = (data.logCSCO.dropna().as_matrix())

p=0
q=0
d=1
pdq=[]
aic=[]

for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(ararray, (p,d,q)).fit()
            x = model.aic
            x1 = (p,d,q)
            
            print (x1, x)
            aic.append(x)
            pdq.append(x1)
        except:
            pass
                        
keys = pdq
values = aic
d = dict(zip(keys, values))
minaic=min(d, key=d.get)

for i in range(3):
    p=minaic[0]
    d=minaic[1]
    q=minaic[2]
print ("Best Model is :", (p,d,q))


#   Use the identified parameters to fit the ARIMA model




ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()





data.loc[1:, 'predict'] = np.exp(ARIMIAmod.predict(typ='levels'))
data.head()





plt.plot(data['CSCO'][1:], label='original')
plt.plot(data.predict[1:], label='predict')
plt.legend()





numofsteps = 10
stepahead = ARIMIAmod.forecast(numofsteps)[0]
ferrors = ARIMIAmod.forecast(numofsteps)[2]
ferrors





data['error'] = (data['CSCO'] - data['predict'])
data['sqrError'] = np.square(data['error'])
data['absError'] = np.abs(data['error'])
data.head()





fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.error, label = "Residual")
plt.title("Residuals of Model", size = 10,)
plt.ylabel("Error", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 2)
plt.plot(data.sqrError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Squared", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 3)
plt.plot(data.absError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Abs", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)





fig, ax = plt.subplots(figsize=(10,4))

plot_acf(data.error, lags = 25, ax = ax)
ax.set_title('ACF Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')





fig, ax = plt.subplots(figsize=(10,4))
plot_acf(data.sqrError, lags = 25, ax = ax)
ax.set_title('ACF Squared Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')





fig, ax = plt.subplots(figsize=(10,4))

plot_pacf(data.error, lags = 25, ax = ax)
ax.set_title('Errors PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')





fig, ax = plt.subplots(figsize=(10,4))

plot_pacf(data.sqrError, lags = 25, ax = ax)
ax.set_title('Errors Squared PACF')
ax.set_xlabel('Lags')
ax.set_ylabel('PACF')


#   Now, to run the ARIMA model on test data




test = test.reset_index()
test_logCSCO = list(test.logCSCO.values)

ararray = list(data.logCSCO.values)
test_predict = []
for i in range(99):
    print(i)
    ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()
    test_predict.append(ARIMIAmod.forecast(1)[0])
    ararray.append(test_logCSCO[i])
    
plt.plot(test_logCSCO)
plt.plot(np.array(test_predict).ravel())





plt.figure(figsize=(10,4))
plt.plot(data.logCSCO, label='original')
plt.plot(data.logCSCO.rolling(300).mean(), label='rolling_mean')
plt.plot(data.logCSCO.rolling(5000).mean(), label='rolling_mean')
plt.legend()





plt.figure(figsize=(10,4))
plt.plot(data.logCSCO.rolling(300).std())





data.head()







def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

print("MFE = ", mean_forecast_err(data.CSCO, data.predict))
print("MAE = ", mean_absolute_err(data.CSCO, data.predict))


#   Mean Forecast error is practically zero; this shows that ARIMA is a great fit
# # ----------------------------------

#   Now, for the FIFTH stock in portfolio : EBAY




df['DATE_F'] = df.DATE.apply(lambda x:time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
df['logEBAY']= np.log(df['EBAY'])
df['diflogEBAY'] = df['logEBAY'] - df['logEBAY'].shift(periods=-1)
df = df.dropna()





# Splitting the data into 2 - for model fit and for test
data = df[['DATE','DATE_F','EBAY','logEBAY','diflogEBAY']][:-100]
test = df[['DATE','DATE_F','EBAY','logEBAY','diflogEBAY']][-100:]





print("Size of data ", data.shape)
print("Size of test ", test.shape)





# Plotting the series and its transformation to check for stationarity

fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.EBAY, label = "EBAY Price")
plt.title("Level Closing Price", size = 20,)
plt.ylabel("Price in Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(data.logEBAY, label = 'Log of EBAY', color = 'r')
plt.title("Log of Closing Price", size = 20,)
plt.ylabel("Price in Log Dollars", size = 10)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot( data.diflogEBAY, label = '1st Difference of Log of EBAY', color = 'g')
plt.title("Differenced Log of Closing Price", size = 20,)
plt.ylabel("Differenced Closing Price", size = 10)
plt.xlabel('Day', size = 10)
plt.legend()





#Perform Dickey-Fuller test:
print ('Results of Dickey-Fuller Test:')
dftest = adfuller(data.diflogEBAY, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)


#   Since p-value = 0, this log-first-difference series is STATIONARY; we can now proceed with ARIMA modeling




# Check for autocorrelation
sm.stats.durbin_watson(df['diflogEBAY'].dropna())





# DW is close to 2, so there is minimum autocorrelation

# Lets plot ACF and PACF plots

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['diflogEBAY'].values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['diflogEBAY'], lags=40, ax=ax2)





# To systematically find the ARIMA model parameters

ararray = (data.logEBAY.dropna().as_matrix())

p=0
q=0
d=1
pdq=[]
aic=[]

for p in range(3):
    for q in range(3):
        try:
            model = ARIMA(ararray, (p,d,q)).fit()
            x = model.aic
            x1 = (p,d,q)
            
            print (x1, x)
            aic.append(x)
            pdq.append(x1)
        except:
            pass
                        
keys = pdq
values = aic
d = dict(zip(keys, values))
minaic=min(d, key=d.get)

for i in range(3):
    p=minaic[0]
    d=minaic[1]
    q=minaic[2]
print ("Best Model is :", (p,d,q))





p, d, q


#   Use the identified parameters to fit the ARIMA model




ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()





data.loc[1:, 'predict'] = np.exp(ARIMIAmod.predict(typ='levels'))
data.head()





plt.plot(data['EBAY'][1:], label='original')
plt.plot(data.predict[1:], label='predict')
plt.legend()





numofsteps = 10
stepahead = ARIMIAmod.forecast(numofsteps)[0]
ferrors = ARIMIAmod.forecast(numofsteps)[2]
ferrors





data['error'] = (data['EBAY'] - data['predict'])
data['sqrError'] = np.square(data['error'])
data['absError'] = np.abs(data['error'])
data.head()





fig, ax = plt.subplots(figsize=(10,4))

plt.subplot(3, 1, 1)
plt.plot(data.error, label = "Residual")
plt.title("Residuals of Model", size = 10,)
plt.ylabel("Error", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 2)
plt.plot(data.sqrError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Squared", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)

plt.subplot(3, 1, 3)
plt.plot(data.absError, label = 'Residual Squared', color = 'r')
plt.title("Residuals of Model Abs", size = 10,)
plt.ylabel("Error Squared", size = 10)
plt.xlabel('Month', size = 10)





fig, ax = plt.subplots(figsize=(10,4))

plot_acf(data.error, lags = 25, ax = ax)
ax.set_title('ACF Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')





fig, ax = plt.subplots(figsize=(10,4))

plot_acf(data.sqrError, lags = 25, ax = ax)
ax.set_title('ACF Squared Error')
ax.set_xlabel('Lags')
ax.set_ylabel('ACF')


#   Now, to run the ARIMA model on test data




test = test.reset_index()
test_logEBAY = list(test.logEBAY.values)

ararray = list(data.logEBAY.values)
test_predict = []
for i in range(99):
    print(i)
    ARIMIAmod = ARIMA(ararray, (p,d,q)).fit()
    test_predict.append(ARIMIAmod.forecast(1)[0])
    ararray.append(test_logEBAY[i])
    
plt.plot(test_logEBAY)
plt.plot(np.array(test_predict).ravel())





plt.figure(figsize=(10,4))
plt.plot(data.logEBAY, label='original')
plt.plot(data.logEBAY.rolling(300).mean(), label='rolling_mean')
plt.plot(data.logEBAY.rolling(5000).mean(), label='rolling_mean')
plt.legend()





plt.figure(figsize=(10,4))
plt.plot(data.logEBAY.rolling(300).std())





data.head()







def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

def mean_absolute_err(y, yhat):
    return np.mean((np.abs(y.sub(yhat).mean()) / yhat)) # or percent error = * 100

print("MFE = ", mean_forecast_err(data.EBAY, data.predict))
print("MAE = ", mean_absolute_err(data.EBAY, data.predict))


#   Mean Forecast error is practically zero; this shows that ARIMA is a great fit

#   In this project, ARIMA (p, d, q) models have been successfully demonstrated for the 5 stocks - AAPL, ADP, CBOE, CSCO, EBAY
