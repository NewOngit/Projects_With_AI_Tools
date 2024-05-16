import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
from math import sqrt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from numpy import concatenate
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.layers import InputLayer,LSTM,Dense,Conv1D, MaxPooling1D, Flatten, Dense, Input,Dropout
from tensorflow.keras.models import Sequential

###Fetching the file 
#df_ALL=pd.read_excel("Weather_Data_14-11-30.xlsx")
df_ALL=pd.read_excel("https://data.london.gov.uk/download/photovoltaic--pv--solar-panel-energy-generation-data/b4a7e790-8cb8-451c-b828-c4c5d8445705/Weather%20Data%202014-11-30.xlsx")

###Columns 
df_ALL.columns
########Collecting the only YMCA Sites Dataset#############
df_YMCA=df_ALL[df_ALL['Site']=="YMCA"]
##Dropping the columns

ls=['Site',  'Month', 'Time', 'Hour','WindDir', 'WindRun', 'HiSpeed','HiDir','Bar','Rain', 'RainRate','InEMC',
       'InAirDensity', 'ET', 'WindSamp', 'WindTx', 'ISSRecept', 'ArcInt']
df_YMCA_M=df_YMCA.drop(ls,axis=1)

SolarEnergy_Data=df_YMCA_M.loc[:,'SolarEnergy']

SolarEnergy=SolarEnergy_Data.values.reshape(SolarEnergy_Data.values.shape[0],1)

new_df_YMCA_M=df_YMCA_M.drop(['SolarEnergy'],axis=1)

ls1=new_df_YMCA_M.columns

df_lon=pd.DataFrame(np.concatenate((new_df_YMCA_M,SolarEnergy),axis=1))

New_Cols=(ls1).to_list()

New_Cols.append("SolarEnergy")
len(New_Cols)

df_lon.columns=New_Cols

#df_lon=pd.read_csv("D:/new_Major_Final.csv",index_col=0)
#df_lon=pd.read_csv("D:/new_Major.csv",index_col=0)

df_lon.head()

#df_lon.describe(include='all',datetime_is_numeric=True) 

(df_lon['THSWIndex']=='---').sum()

col_=df_lon.columns
col_.size

###################################Function definition for feature selection and change the proper datatypes##################
def feature_Selection(Dataframe,col_):  
    for _ in col_:
        if (Dataframe[_]=='---').sum()<len(Dataframe[_])/2:
            df_=Dataframe.replace({_: r'^--.$'}, {_: np.NaN}, regex=True)
        else:
            Dataframe=Dataframe.drop([_],axis=1)
    return df_

df_=feature_Selection(df_lon,df_lon.columns)
df_

print("The NaN elements are :")
print(df_lon.isna().sum())
explodes=(0,0.15)
plt.figure(0)
if df_lon.isna().sum().sum()!=0:
   plt.pie(df_lon.isna().value_counts(),explode=explodes,startangle=0,colors=["firebrick","indianred"],
    labels=['Non NaN elements',str("%0.2f"%(df_lon.isnull().sum().sum()*100/len(df_lon.Wind_Speed)))+'% NaN elements'],textprops={'fontsize':'20'}) 
else :
    plt.pie(df_lon.isna().value_counts(),startangle=0,colors=["firebrick","indianred"],
    labels=['There is no NaN element present'],textprops={'fontsize':'20'}) 

    
#Dropping the inappropriate type columns from the dataset
df_=df_.drop(['Date'],axis=1)

df_.columns

df_=df_.astype(float)


###################################Line Plotting of the Dataset#############################
def line_plot(Dataframe):
    cols=Dataframe.columns
    df_size=int(len(Dataframe))
    fig,axs=plt.subplots(len(cols),2)
    fig.set_figwidth(20)
    fig.set_figheight(120)
    i=0
    for _ in cols:
        axs[i,0].set_title('Whole Dataset of '+str(_),fontsize=20)
        axs[i,0].plot(Dataframe[_])
        axs[i,1].set_title('5% Dataset of '+str(_),fontsize=20)
        axs[i,1].plot(Dataframe[_][df_size-int(0.05*df_size):],'tab:orange')
        i=i+1
        

line_plot(df_)

#Function definition for the density plotting of Dataset 
def density_plot(Dataframe):
 cols=Dataframe.columns
 #fig,axs=plt.subplots(len(cols),1)
 i=0
 for _ in cols:
     plt.figure(i)
     Dataframe[_].plot.density(color="blue")
     plt.title(_)
     i=i+1 

density_plot(df_)


def heatmap(Dataframe):
    df=Dataframe.corr()
    sns.heatmap(df,cmap='RdBu',vmin=-1,vmax=1,annot=True)

heatmap(df_)

#Function definition for the density plotting of Dataset 
def scatter_plot(Dataframe,target):
    col=Dataframe.columns
    fig,axs=plt.subplots(len(col)-1)
    fig.set_figwidth(15)
    fig.set_figheight(90)
    fig.tight_layout(pad=5.0)
    i=0
    for _ in col: 
         if target!=_:
             axs[i].set_title('Scatter Plot between '+target+" and "+str(_),fontsize=20)
             axs[i].scatter(Dataframe[target],Dataframe[_])
             axs[i].set_ylabel(_,fontsize=18)
             axs[i].set_xlabel(target,fontsize=18)
             i=i+1
             
    

#!!!!!!!!!!!!!!!!!!!!!!Deprecated Version
"""
#Function definition for the density plotting of Dataset 
def scatter_plot(Dataframe,target):
    col=Dataframe.columns
    i=0
    for _ in col: 
         if target!=_:
             plt.figure(i)
             plt.scatter(Dataframe[target],Dataframe[_],color="blue")
             plt.ylabel(_)
             plt.xlabel(target)
             i=i+1
   """          
    
#####Plotting the scatter plot
scatter_plot(df_,df_.columns[len(df_.columns)-1])


