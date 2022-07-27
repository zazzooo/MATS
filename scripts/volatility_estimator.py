import pandas as pd
import numpy as np
import yfinance as yf
from bgdataloader import polygon_fetch as pf

#utiliteis for notebooks
def create_dataframe_high_low(list_securities):
    df = pd.DataFrame()
    for security in list_securities:
        SEC = yf.Ticker(security)
        hist = SEC.history(period="max")
        df[security + '_low'] = hist['Low']
        df[security + '_high'] = hist['High']
        df.dropna(inplace = True) #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    return df

def create_dataframe_high_low_polygon(list_securities, start_date, end_date):
    df = pd.DataFrame()
    for security in list_securities:
        pc = pf.PolygonClient()
        client = pc.get_client()
        hist = pc.get_equity_bar_data(security, 1, "day",start_date, end_date)
        df[security + '_low'] = hist['low']
        df[security + '_high'] = hist['high']
        df.dropna(inplace = True) #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    return df

def compute_park_volatility(df_high_low, list_securities, window):
    '''
    input
    df = pandas dataframe with columns named: "security_low" and "security_high" (e.g. SSO_low SSO_high)
    list_securities = list of securities name
    window = int, number of days the index has to consider

    output
    df_output = pandas dataframe with securities as columns and parkinson volat estimator as rows

    '''


    k = np.sqrt(1/(4*window*np.log(2)))
    df_output = pd.DataFrame()

    for security in list_securities:
        #crating array to store the indexes (one per day)
        array_value = np.array([])
        for i in range(len(df_high_low) - window):

            #creating array where storing the log((High_price/Low_price)^2)
            array_day_value = np.array([])
            for j in range(window):
                #array_day_value = np.array([])

                #appending the value (one for each window day)
                array_day_value = np.append(array_day_value, np.log(df_high_low[security + '_high'][i+j] / df_high_low[security + '_low'][i+j])**2)

                # len(array_day_value) = window
            square_sum_value = np.sqrt(np.sum(array_day_value))
            array_value = np.append(array_value, k*square_sum_value)

        #dict_security_park_indx[security] = array_value
        df_output[security] = array_value
    return df_output #dict_security_park_indx,

def index_df(df_output, data, window):
    #just add the index to the dataframe
    data_copy = data.copy()
    data_copy.drop(data_copy.head(window).index, inplace=True) # drop last n rows

    return df_output.set_index(data_copy.index)

def dataframe_strd_dev(data, window):
    '''
    input
    data = pandas dataframe, index: timestamp; column: name security; value: close price (or what ever you need).
     window = number of day you want ton compute the standard deviation on.

     output
     df_output = pandas dataframe with security as column, and standard deviation as value
    '''
    df_output = pd.DataFrame()

    list_securities = data.columns
    for security in list_securities:
        #crating array to store the indexes (one per day)
        array_value = np.array([])
        for i in range(len(data) - window):
            array_value = np.append(array_value, np.std(data[security][i:i+window]))

        # assign new array as column of the df
        df_output[security] = array_value

    return df_output

def create_dataframe_ohlc(list_securities):
    df = pd.DataFrame()
    for security in list_securities:
        SEC = yf.Ticker(security)
        hist = SEC.history(period="max")
        df[security + '_open'] = hist['Open']
        df[security + '_high'] = hist['High']
        df[security + '_low'] = hist['Low']
        df[security + '_close'] = hist['Close']
        df.dropna(inplace = True) #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    return df

def create_dataframe_ohlc_polygon(list_securities, start_date, end_date):
    df = pd.DataFrame()
    for security in list_securities:
        pc = pf.PolygonClient()
        client = pc.get_client()
        hist = pc.get_equity_bar_data(security, 1, "day",start_date, end_date)
        df[security + '_open'] = hist['open']
        df[security + '_high'] = hist['high']
        df[security + '_low'] = hist['low']
        df[security + '_close'] = hist['close']
        df.dropna(inplace = True) #if this line the data are only from 2010 otyherwise some columns arrive until 2006
    return df

def compute_satchell_volatility(df_ohlc, list_securities, window):
    '''
    input
    df_ohlcv = pandas dataframe, index: timestamp; column: name security + ohlc; value: ohlc price (or what ever you need).
    list_securities = list of securiyties used
    window = number of day you want ton compute the standard deviation on.

     output
     df_output = pandas dataframe with security as column, and satchell volatility as value (per security)
    '''
    df_output = pd.DataFrame()

    for security in list_securities:
        #crating array to store the indexes (one per day)
        array_value = np.array([])
        for i in range(len(df_ohlc) - window):

            #creating array where storing the log((High_price/Low_price)^2)
            array_day_value = np.array([])
            for j in range(window):
                #array_day_value = np.array([])

                #appending the value (one for each window day)
                array_day_value = np.append(array_day_value, np.log((df_ohlc[security + '_high'][i+j] / df_ohlc[security + '_close'][i+j]))*np.log((df_ohlc[security + '_high'][i+j] / df_ohlc[security + '_open'][i+j]))
                                                            + np.log((df_ohlc[security + '_low'][i+j] / df_ohlc[security + '_close'][i+j]))*np.log((df_ohlc[security + '_low'][i+j] / df_ohlc[security + '_open'][i+j])))
                # len(array_day_value) = window
            square_sum_value = np.sqrt((1/window)*np.sum(array_day_value))
            array_value = np.append(array_value,square_sum_value)

        #dict_security_park_indx[security] = array_value
        df_output[security] = array_value
    return df_output #dict_security_park_indx,
