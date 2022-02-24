# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:58:57 2022

@author: sengu

Experimenting with first change point detection method 

1. Segmentation by Sliding window - online version
    
    a. Naive sliding window extractor  ----- In progress
    
2. Segmentation by Top-down approach ---- Later
3. segmentation by Botton-up approach ---- Later
4. Dataset - Amazon dataset
    
    a. Do not pre-process
    b. Pre-process the data
    c. apply on another dataset/series
5. SWAB (Sliding window and bottom-up approach)


This series clearly is a non-stationary time series.


Pre_processing:
1. We do not apply any pre_processing and work on the raw time series
2. We apply pre_processing and work on the pre_processed series


Model selection:
1. Piecewise Linear
2. Piecewise polynomial
3. Piecewise spline
4. Linear Regression
5. Polynomial function
6. Filters (Kalman etc)


"""

from Code_Data.PreProcessing import PreProcessing
from Code_Data.Visualization import Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as it
from scipy import optimize
import pwlf

########## using a package for sliding window segmentation as reviewed in the literature:
#https://dmbee.github.io/seglearn/user_guide.html

import seglearn
from seglearn.base import TS_Data

pre_process = PreProcessing()



"""
@argument - filter_criteria (Takes boolean - True/False)
@out - returns a dataframe (filtered vs not_filtered). 

What filter to use?

1. Differencing the data
2. Any smoothing techniques applied on the whole dataset
"""
def pre_process_data(filter_criteria):
    if filter_criteria == False:
       data_in = pre_process.import_file()
       data_in = data_in[['Date','Close']]
       data_in = data_in.set_index('Date')
       data_in_days = pd.DataFrame()
       data_in_weeks = pd.DataFrame()
       data_in_days['Close'] = data_in['2019-10-01':'2021-09-30'].resample('D').sum()
       data_in_days = data_in_days[data_in_days.Close != 0]
       
       data_in_weeks['Close'] = data_in['2019-10-01':'2021-09-30'].resample('W').mean()
       data_in_weeks = data_in_weeks.dropna(axis="rows")
       
       fig, ax = plt.subplots(figsize=(20, 6))
       ax.plot(data_in_days,marker='.', linestyle='-', linewidth=0.5, label='Daily')
       ax.plot(data_in_days.resample('W').mean(),marker='o', markersize=8, linestyle='-', label='Weekly')
       ax.axhline(y=data_in_days['Close'].mean(), color='red', linestyle='--')
       ax.axhline(y=data_in_weeks['Close'].mean(), color='black', linestyle='--')
       ax.set_ylabel('Closing Price')
       ax.legend();
       plt.show()
       return data_in_days,data_in_weeks
    
    ####################### apply pre_processing techniques on the series
    if filter_criteria == True:
        data_in = pre_process.import_file()
        data_in = data_in[['Date','Close']]
        data_in = data_in.set_index('Date')
        data_in_days = pd.DataFrame()
        data_in_weeks = pd.DataFrame()
        data_in_days['Close'] = data_in['2019-10-01':'2021-09-30'].resample('D').sum()
        data_in_days = data_in_days[data_in_days.Close != 0]
        
        data_in_weeks['Close'] = data_in['2019-10-01':'2021-09-30'].resample('W').mean()
        data_in_weeks = data_in_weeks.dropna(axis="rows")
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(data_in_days,marker='.', linestyle='-', linewidth=0.5, label='Daily')
        ax.plot(data_in_days.resample('W').mean(),marker='o', markersize=8, linestyle='-', label='Weekly')
        ax.axhline(y=data_in_days['Close'].mean(), color='red', linestyle='--')
        ax.axhline(y=data_in_weeks['Close'].mean(), color='black', linestyle='--')
        ax.set_ylabel('Closing Price')
        ax.legend();
        plt.show()
        return data_in_days,data_in_weeks
    


""" Naive Sliding window implementation
@argument - filter_criteria (Takes boolean - True/False)
@out - returns list of Closing price using the iterator object for different sliding windows we want

"""
def naive_extract_windows(df, length, step=1):
    x = df.Close
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step*length) for stream, i in zip(streams, it.count(step=step))])



""" Converting the sliding window arrays as feature sets
@argument - Null
@output - Here the sliding window arrays obtained from naive_extract_windows function, is a list of tupples

------------1) converted into a dataframe where every arrays of sliding window values are converted into feature sets
------------2) we transpose the dataframe as it is a time series data and we want to maintain the temporal sequence of the observation
"""
def convert_slidingwindows_features():
    ######## considering a sliding window of 5 days (we can test later with other window size)
    data_in_days,data_in_weeks = pre_process_data(False)
    windows_segment_days = list(naive_extract_windows(data_in_days,5)) ### eliminates the last 2 elements because we set step = 1
    newdf = pd.DataFrame(windows_segment_days).T
    
    df_list = [newdf]
    for i , newdf in enumerate(df_list):
        newdf.columns = [str(col_name)+'F{}'.format(i) for col_name in newdf.columns]
    newdf.reset_index(inplace = True)
    return newdf 


""" Test visualization of few time series decomposed
@argument - Null
@output - time series for every 5 observations divided into segments (5 series being decomposed)
"""
def some_test_visualization():
    newdf = convert_slidingwindows_features()
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(newdf['0F0'],marker='.', linestyle='-', linewidth=0.5, label='feature1')
    ax.plot(newdf['1F0'],marker='.', linestyle='-', linewidth=0.5, label='feature2')
    ax.plot(newdf['2F0'],marker='.', linestyle='-', linewidth=0.5, label='feature3')
    ax.plot(newdf['3F0'],marker='.', linestyle='-', linewidth=0.5, label='feature4')
    ax.plot(newdf['4F0'],marker='.', linestyle='-', linewidth=0.5, label='feature5')
    ax.set_ylabel('Closing Price')
    ax.legend();
    plt.show()

some_test_visualization()


""" Applying customised segmentation to perform approximation on every 93 features obtained

Descrition: we develop a custom segmentation function over a naive sliding window method


Linear function--------------------- (Parametric approach)
1. fitting piecewise linear  - piecewise linear regression
2. fitting piecewise polynomial functions - piecewise polynomial function

Cost function----------------------- (Parametric approach)
1. MSE
2. Distance metric like Mahalanobis method

3. fitting interpolation methods






"""


"""
Piecewise Linear function (POC on the first and second segment, we can then replicate on all the segments)
"""
def piecewise_linear(x, y):
    pwlf_fit = pwlf.PiecewiseLinFit(x, y)
    breaks = pwlf_fit.fit(2)
    print(breaks)
    x_hat = np.linspace(x.min(), x.max(), 100)
    y_hat = pwlf_fit.predict(x_hat)
    return x_hat,y_hat



"""
Linear regression on every segments (POC on the first and second segment, we can then replicate on all the segments)
"""
def apply_customized_segmentation(ftype):
    if ftype == "piecewise":
        newdf = convert_slidingwindows_features()
        x = np.array(newdf.index)
        y = np.array(newdf['0F0'])
        x_hat,y_hat = piecewise_linear(x, y)  ############ piecewise function call
        
        plt.figure()
        plt.plot(x, y, 'o')
        plt.plot(x_hat, y_hat, '-')
        plt.show()
        return x,y,x_hat,y_hat
    
    #if ftype == "LR":
        
        

x,y,x_hat,y_hat = apply_customized_segmentation("piecewise")

    
    
    




















        