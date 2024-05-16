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

###################################Function definition for feature selection and change the proper datatypes##################
def feature_Selection(Dataframe,col_):  
    for _ in col_:
        if (Dataframe[_]=='---').sum()<len(Dataframe[_])/2:
            df_=Dataframe.replace({_: r'^--.$'}, {_: np.NaN}, regex=True)
        else:
            Dataframe=Dataframe.drop([_],axis=1)
    return df_

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

def heatmap(Dataframe):
    df=Dataframe.corr()
    sns.heatmap(df,cmap='RdBu',vmin=-1,vmax=1,annot=True)


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
    
############################Function Definition for finding the R2 Score of Liera Regression##################
def linear_reg(train_X,train_y,test_X,test_y):
    linearReg_clf = LinearRegression()
    linearReg_clf.fit(train_X,train_y)
    X_pred_linearReg=linearReg_clf.predict(train_X)
    y_pred_linearReg=linearReg_clf.predict(test_X)
    score_linearReg = 100*linearReg_clf.score(test_X,test_y)
    print(f'Linear Regressor Model R2 score = {score_linearReg:4.4f}%')
    return score_linearReg,y_pred_linearReg,X_pred_linearReg

############################Function Definition for finding the R2 Score of Randon Forest##################
def random_forest(train_X,train_y,test_X,test_y):
    RandForest = RandomForestRegressor()
    RandForest.fit(train_X,train_y)
    y_pred_RandForest = RandForest.predict(test_X)
    X_pred_RandForest=RandForest.predict(train_X)
    R2_Score_RandForest = round(r2_score(y_pred_RandForest,test_y) * 100, 2)
    print("Random Forest R2 Score : ",R2_Score_RandForest,"%")
    return R2_Score_RandForest,y_pred_RandForest, X_pred_RandForest

############################Function Definition for finding the R2 Score of Decision Tree Regressor##################
def decision_tree(train_X,train_y,test_X,test_y):
    dtr = DecisionTreeRegressor()
    dtr.fit(train_X,train_y)
    y_pred_DecTreeReg = dtr.predict(test_X)
    X_pred_DecTreeReg = dtr.predict(train_X)
    R2_Score_DecTreeReg = round(r2_score(y_pred_DecTreeReg,test_y) * 100, 2)
    print("Decision Tree R2 Score : ",R2_Score_DecTreeReg,"%")
    return R2_Score_DecTreeReg,y_pred_DecTreeReg,X_pred_DecTreeReg

############################Function Definition for finding the R2 Score of SVM Regressor##################
def support_vector_machine(train_X,train_y,test_X,test_y):
    regr = svm.SVR()
    regr.fit(train_X, train_y)
    y_pred_svmReg = regr.predict(test_X)
    X_pred_svmReg = regr.predict(train_X)
    R2_Score_svmReg = round(r2_score(y_pred_svmReg,test_y) * 100, 2)
    print("Support Vector Machine R2 Score : ",R2_Score_svmReg,"%")
    return R2_Score_svmReg,y_pred_svmReg,X_pred_svmReg

#############################Best Model For The Dataset#####################
def best_model(dict):
    dd=pd.DataFrame(dict.items())
    dd.columns=['Model','R2_Score']
    dd1=dd.sort_values(by='R2_Score',ascending=False).reset_index(drop=True)
    print("Best Model is : "+str(dd1['Model'][0]))
    return  dd1

def line_plot_Preicted_versus_Actual_Dataset(predicted,actual,model_name):
 i=0
 fig,ax=plt.subplots(nrows=1,ncols=1)
 fig.suptitle(model_name+" Prediction Evaluation")
 ax.plot(predicted,label='Predicted Data')
 ax.plot(actual,label='Original Data')
 ax.set_title("")
 ax.set_xlabel("")
 ax.set_ylabel("power generated")
 plt.legend()
 plt.show()
 return

def Accuracy_Measurements1(Df_original,predicted,model_name):
    dict={model_name:(round((mean_absolute_error(Df_original,predicted))*100,5),round((mean_squared_error(Df_original,predicted))*100,5),round((sqrt(mean_squared_error(Df_original,predicted)))*100,5))}
    return dict

def model_performace(model_history):
 plt.plot(model_history.history['loss'],label="training loss")
 plt.plot(model_history.history['val_loss'],label="validation loss")
 plt.legend()
 plt.show()

 def transform_from_res(predicted,input_data):
    df_1=pd.DataFrame(input_data)
    input_data_1=pd.DataFrame(test_predict_1)
    reslutant_=df_1+input_data_1
    result_ser=reslutant_.values
    test_predict_H_1=result_ser.reshape(result_ser.shape[0],1)
    return test_predict_H_1
 
 def line_plot_Preicted_versus_Actual_Dataset(predicted,actual,model_name):
    i=0
    fig,ax=plt.subplots(nrows=1,ncols=1)
    fig.suptitle(model_name+" Prediction & Evaluation")
    ax.plot(predicted,label='Predicted Data')
    ax.plot(actual,label='Original Data')
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("power generated")
    plt.legend()
    plt.show()
    return
 
 def Accuracy_Measurements_H(Df_original,predicted,model_name):
    dict={model_name:(round((mean_absolute_error(Df_original,predicted))*100,5),round((mean_squared_error(Df_original,predicted))*100,5),round((sqrt(mean_squared_error(Df_original,predicted)))*100,5))}
    return dict
 

def line_plot_Preicted_versus_Actual_Dataset(predicted,actual,model_name):
    i=0
    fig,ax=plt.subplots(nrows=1,ncols=1)
    fig.suptitle(model_name+" Prediction Evaluation")
    ax.plot(predicted,label='Predicted Data')
    ax.plot(actual,label='Original Data')
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("power generated")
    plt.legend()
    plt.show()
    return

def Accuracy_Measurements_DL(Df_original,predicted,model_name):
    dict={model_name:(round((mean_absolute_error(Df_original,predicted))*100,5),round((mean_squared_error(Df_original,predicted))*100,5),round((sqrt(mean_squared_error(Df_original,predicted)))*100,5))}
    return dict

def df_X_y(df,window_size):
    df_as_np=df.to_numpy()
    X=[]
    y=[]
    for i in range(len(df_as_np)-window_size):
        row=[r for r in df_as_np[i:i+window_size]]
        val=df_as_np[i+window_size][-1]
        X.append(row)
        y.append(val)
    return np.array(X), np.array(y)

##########change the ndArray from 2d to 1D
def twoD_oneD(df):
    df1=pd.DataFrame(df)
    ls= df1[0].to_numpy()
    ls1=[]
    for elem in ls:
        ls1.append(elem)
    return pd.DataFrame(ls1)


def transform_(df):
    d=[]
    #two_=[[[]]]
    for two_ in df:
        for one_ in two_:
            d.append(one_)
            break
    return d
            


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Depracated Version 
"""
def Accuracy_Measurements_H(Df_original,predicted,model_name):
    dict={model_name:(round((mean_absolute_error(Df_original,predicted))*100,5),round((mean_squared_error(Df_original,predicted))*100,5),round((sqrt(mean_squared_error(Df_original,predicted)))*100,5),round((mean_absolute_percentage_error(Df_original,predicted))*100,5))}
    return dict
    """

def Accuracy_Measurements_H(Df_original,predicted,model_name):
    dict={model_name:(round((mean_absolute_error(Df_original,predicted))*100,5),round((mean_squared_error(Df_original,predicted))*100,5),round((sqrt(mean_squared_error(Df_original,predicted)))*100,5))}
    return dict






