# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:37:38 2021

@author: sengu
"""

# For graphing purpose, can change 
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 8
matplotlib.rcParams['ytick.labelsize'] = 8
matplotlib.rcParams['text.color'] = 'k'
import seaborn as sns
import statsmodels.api as sm
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


class Visualization:
    
    def initial_plot(self,df):
        return df.plot()
    
    def hist_plot(self,df):
        sns.set(style="darkgrid")
        fig, axs = plt.subplots(2, 2, figsize=(7, 7))
        sns.histplot(data=df, x="Close", kde=True, color="skyblue", ax=axs[0, 0])
        return fig
        
        
    def visualize_initial_data(self,withfilter,df):
        if withfilter == False:
            fig, ax = plt.subplots(figsize=(20, 6))
            ax.plot(df,marker='.', linestyle='-', linewidth=0.5, label='Daily-NotFiltered')
            ax.plot(df.resample('W').mean(),marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample-NotFiltered')
            ax.plot(df.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample-NotFiltered')
            ax.set_ylabel('Closing Price')
            ax.legend();
            plt.show()
            
        if withfilter == True:
            fig, ax = plt.subplots(figsize=(20, 6))
            ax.plot(df,marker='.', linestyle='-', linewidth=0.5, label='Daily-Filtered')
            ax.plot(df.resample('W').mean(),marker='o', markersize=8, linestyle='-', label='Weekly Mean Resample-Filtered')
            ax.plot(df.resample('M').mean(),marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample-Filtered')
            ax.set_ylabel('Closing Price')
            ax.legend();
            plt.show()
            
    def seasonal_plots(self,df):
        df = pd.DataFrame(df)
        decomposition1 = sm.tsa.seasonal_decompose(df, model="additive")
        fig = decomposition1.plot()
        fig.set_size_inches(10,7)
        return fig
        
    
    def acf_pacf_plots(self,df):
        a= plot_acf(df)
        p = plot_pacf(df)
        return a,p
    
    def adf_test(self,timeseries):
        print ('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        return print(dfoutput)
    
    def kpss_test(self,timeseries):
        print ('Results of KPSS Test:')
        kpsstest = kpss(timeseries, regression='c')
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
        return print(kpss_output)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        