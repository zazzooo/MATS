import pandas as pd
from math import isnan
import pandas_ta as ta

def create_mean_rev_signal(data, long_wind, short_wind):
    '''
    input 
    data: pandas dataframe with prices of securities (name of the column security)
    long_wind: the long window you want for the mean reverse strategy
    short_wind: the short windw you want for the mean reverse strategy
    
    output:
    final_df: pandas df filled with 1 or -1 accordingly to the mean reverse strategy
    df_rolling_mean: df with the means over the different windows for debugging
    '''
    final_df = pd.DataFrame()
    df_rolling_mean = pd.DataFrame()
    for sec in data.columns:
        sec_df = pd.DataFrame(columns = [sec])
        long_wind_df = data[sec].rolling(long_wind).mean() #creating the long window df
        df_rolling_mean[sec + '_long'] = long_wind_df #for debugging
        short_wind_df = data[sec].rolling(short_wind).mean() #creating the short window df
        df_rolling_mean[sec + '_short'] = short_wind_df #for debugging
        sec_df[sec] = (short_wind_df - long_wind_df) #subtracting the long and the short
        #sec_df[sec] = sec_df[sec].apply(lambda x: 1 if x>0 else -1) 
        final_df[sec] = sec_df[sec]
    final_df = final_df.applymap(lambda x: 1 if x>0 else -1, na_action = 'ignore') #subsitute positive value with 1 and negative with a -1
    return final_df, df_rolling_mean

def filter_dataframe_hl(df_tot, list_securities):
    df_output = pd.DataFrame()
    for security in list_securities:
        df_output[security + '_high'] = df_tot[security]['High']
        df_output[security + '_low'] = df_tot[security]['Low']
        #df_output.dropna(inplace = True)
    return df_output

def filter_dataframe_holc(df_tot, list_securities):
    df_output = pd.DataFrame()
    for security in list_securities:
        df_output[security + '_open'] = df_tot[security]['Open']
        df_output[security + '_high'] = df_tot[security]['High']
        df_output[security + '_low'] = df_tot[security]['Low']
        df_output[security + '_close'] = df_tot[security]['Close']
        #df_output.dropna(inplace = True)
    return df_output

def number_nan(row):
    count = 0
    for i in range(len(row)):
        if isnan(row[i]):
            count += 1
    return len(row)-count

def create_kama_signal_df(data):
    '''
    input 
    data: pandas dataframe with prices of securities (name of the column security)
    
    output:
    final_df: pandas df filled with 1 or -1 accordingly to the kama strategy
    '''
    final_df = pd.DataFrame()
    df_kama = pd.DataFrame()
    for sec in data.columns:
        sec_df = pd.DataFrame(columns = [sec])
        sec_df[sec] = (ta.overlap.kama(data[sec]) - data[sec]) #subtracting the long and the short
        #sec_df[sec] = sec_df[sec].apply(lambda x: 1 if x>0 else -1) 
        final_df[sec] = sec_df[sec]
    final_df = final_df.applymap(lambda x: 1 if x>0 else -1, na_action = 'ignore') #subsitute positive value with 1 and negative with a -1
    return final_df