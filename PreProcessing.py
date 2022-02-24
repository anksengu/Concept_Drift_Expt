# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:37:29 2021
@author: sengu
Expt 3b & 3d : Amazon Stock Price to compare SARIMA vs Prophet
"""

import pandas as pd
import numpy as np
import warnings
import itertools
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.stattools import adfuller
import os
from statsmodels.tsa.arima_model import ARIMA, ARMA
import warnings
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit

#!pip install pmdarima
import pmdarima as pm
from pmdarima import arima
from pmdarima import model_selection
from pmdarima import pipeline
from pmdarima import preprocessing
from pmdarima.datasets._base import load_date_example
import holidays as holidays
from pathlib import Path
from scipy.stats import boxcox
import matplotlib
import matplotlib.pyplot as plt

#os.chdir(r"C:\Users\sengu\OneDrive\Documents\PhD UCD 2021\Experiments\Expt3_Comparison\ConsolidatedExpt\PhDExperiments\Code&Data")

os.chdir(r"C:\Users\sengu\OneDrive\Documents\PhD UCD 2021\Experiments\Expt3_Comparison\ConsolidatedExpt\PhDExperiments\Code_Data")

class PreProcessing:
    
    def import_file(self):
        df = pd.read_csv("AMZN.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    """
    Function: which resamples the day level of data to weeks and months
    """
    def resample_strategy(self,aggregate_level):
        df = PreProcessing.import_file(self)
        df = pd.DataFrame(df)
        df = df[['Date', 'Close']]
        if aggregate_level == "days":
            newdf = df.copy()
            newdf = newdf.set_index('Date')
            newdf = newdf['2019-10-01':'2021-09-30'].resample('D').sum()
            return newdf
            
        if aggregate_level == "weeks":
            newdf = df.copy()
            newdf = newdf.set_index('Date')
            newdf = newdf['2019-10-01':'2021-09-30'].resample('W').mean()
            return newdf


    """
    Function: which filters the data based on the with filter flag
    @withfilter flag = False (Returns the original dataset)
    @withfilter flag = True (Returns the fitered dataset)
    @aggregate level = aggregate/sample the data at days/weeks level
    This function handles all the filtering & Non-filtering criteria
    
    @if withfilter = True (Does the filtering based on dates)
    @if withfilter = False (Returns the original dataset)
    """
    def prepare_dataset(self,withfilter,agg_level):
        if withfilter == False and agg_level == "days":
            newdf = PreProcessing.resample_strategy(self,"days")
            return newdf
        
        elif withfilter == False and agg_level == "weeks":
            newdf = PreProcessing.resample_strategy(self,"weeks")
            return newdf
        
        elif withfilter == True and agg_level == "days":
            newdf = PreProcessing.resample_strategy(self,"days")
            newdf = newdf.loc['2020-07-01':'2021-09-30']
            return newdf
            
        elif withfilter == True and agg_level == "weeks":
            newdf = PreProcessing.resample_strategy(self,"weeks")
            newdf = newdf.loc['2020-07-01':'2021-09-30']
            return newdf
    
    """
    Function: which extracts only the Close column
    @withfilter flag = False (Returns the original dataset)
    @withfilter flag = True (Returns the fitered dataset)
    @aggregate level = aggregate/sample the data at days/weeks level
    """
    def return_dataset(self,withfilter,agg_level):
        if withfilter == False and agg_level == "days":
            df = PreProcessing.prepare_dataset(self,False,"days")
            return df
            
        if withfilter == False and agg_level == "weeks":
            df = PreProcessing.prepare_dataset(self,False,"weeks")
            df.dropna(inplace = True)
            return df
            
        if withfilter == True and agg_level == "days":
            df = PreProcessing.prepare_dataset(self,True,"days")
            return df
            
        if withfilter == True and agg_level == "weeks":
            df = PreProcessing.prepare_dataset(self,True,"weeks")
            return df
        
    
    """
    Function: which detrends the data
    @withfilter flag = False (Returns the original dataset)
    @withfilter flag = True (Returns the fitered dataset)
    @aggregate level = aggregate/sample the data at days/weeks level
    @lags = considering the number of lags to detrend the data i.e. remove the trend or difference the data
    """
    def detrend_dataset(self,lags,agg_level,withfilter,withdetrend):
        if withfilter == False and withdetrend == True:
            if agg_level == "weeks":
                df = PreProcessing.return_dataset(self,False,"weeks")
                df_t_adj =  df - df.shift(lags)
                df_t_adj = df_t_adj.dropna()
                plt.figure(figsize=(12,4))
                plt.plot(df_t_adj, label='Weeks not filtered and detrended')
                plt.legend(loc='best')
                plt.title('Trend Difference')
                plt.show()
            if agg_level == "days":
                df = PreProcessing.return_dataset(self,False,"days")
                df_t_adj =  df - df.shift(lags)
                df_t_adj = df_t_adj.dropna()
                plt.figure(figsize=(12,4))
                plt.plot(df_t_adj, label='Days not filtered and detrended')
                plt.legend(loc='best')
                plt.title('Trend Difference')
                plt.show()
        if withfilter == True and withdetrend == True:
            if agg_level == "weeks":
                df = PreProcessing.return_dataset(self,True,"weeks")
                df_t_adj =  df - df.shift(lags)
                df_t_adj = df_t_adj.dropna()
                plt.figure(figsize=(12,4))
                plt.plot(df_t_adj, label='Weeks filtered and detrended')
                plt.legend(loc='best')
                plt.title('Trend Difference')
                plt.show()
            if agg_level == "days":
                df = PreProcessing.return_dataset(self,True,"days")
                df_t_adj = df - df.shift(lags)
                df_t_adj = df_t_adj.dropna()
                plt.figure(figsize=(12,4))
                plt.plot(df_t_adj, label='Days filtered and detrended')
                plt.legend(loc='best')
                plt.title('Trend Difference')
                plt.show()
        # return the original dataset based on the agg_level
        if withfilter == False and withdetrend == False:
            if agg_level == "days":
                df_t_adj = PreProcessing.return_dataset(self,False,"days")
                plt.figure(figsize=(12,4))
                plt.plot(df_t_adj, label='Days not filtered and not detrended')
                plt.legend(loc='best')
                plt.title('Trend Difference')
                plt.show()
            if agg_level == "weeks":
                df_t_adj = PreProcessing.return_dataset(self,False,"weeks")
                plt.figure(figsize=(12,4))
                plt.plot(df_t_adj, label='Weeks not filtered and not detrended')
                plt.legend(loc='best')
                plt.title('Trend Difference')
                plt.show()
        if withfilter == False and withdetrend == True:
            if agg_level == "days":
                df_t_adj = PreProcessing.return_dataset(self,False,"days")
                df_t_adj = df - df.shift(lags)
                df_t_adj = df_t_adj.dropna()
                plt.figure(figsize=(12,4))
                plt.plot(df_t_adj, label='Days not filtered and detrended')
                plt.legend(loc='best')
                plt.title('Trend Difference')
                plt.show()
            if agg_level == "weeks":
               df_t_adj = PreProcessing.return_dataset(self,False,"weeks")
               df_t_adj = df - df.shift(lags)
               df_t_adj = df_t_adj.dropna()
               plt.figure(figsize=(12,4))
               plt.plot(df_t_adj, label='Weeks not filtered and not detrended')
               plt.legend(loc='best')
               plt.title('Trend Difference')
               plt.show()     
        return df_t_adj
    
    

    """
    Function: which detrends the data
    @withfilter flag = False (Returns the original dataset)
    @aggregate level = aggregate/sample the data at days/weeks level
    @lags = considering the number of lags to detrend the data i.e. remove the trend or difference the data
    """
    def transform_data_difference_data(self, agg_level):
        if agg_level == "days":
            data1 = PreProcessing.return_dataset(self, False, "days")
            data_boxcox = pd.Series(boxcox(data1, lmbda=0), index = data1.index)
            
            data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(1), data1.index)
            plt.figure(figsize=(12,4))
            plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')
            plt.legend(loc='best')
            plt.title('After Box Cox transform and differencing - Daily')
            plt.show()
            data_boxcox_diff.dropna(inplace = True)
            return data_boxcox
            
        if agg_level == "weeks" :
            data1 = PreProcessing.return_dataset(self, False, "weeks")
            data_boxcox = pd.Series(boxcox(data1, lmbda=0), index = data1.index)
            
            data_boxcox_diff = pd.Series(data_boxcox - data_boxcox.shift(), data1.index)
            plt.figure(figsize=(12,4))
            plt.plot(data_boxcox_diff, label='After Box Cox tranformation and differencing')
            plt.legend(loc='best')
            plt.title('After Box Cox transform and differencing - Weekly')
            plt.show()
            data_boxcox_diff.dropna(inplace = True)
            return data_boxcox_diff
    
        