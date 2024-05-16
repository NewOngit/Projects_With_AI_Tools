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
###############################IMPORTING USER DEFINED FUNCTIONS
from User_Defined_Functions import  feature_Selection,line_plot,density_plot,heatmap,scatter_plot,linear_reg,random_forest,decision_tree,support_vector_machine,best_model,line_plot_Preicted_versus_Actual_Dataset,Accuracy_Measurements1,model_performace,transform_from_res,line_plot_Preicted_versus_Actual_Dataset,Accuracy_Measurements_H,line_plot_Preicted_versus_Actual_Dataset,Accuracy_Measurements_DL,df_X_y,twoD_oneD,transform_,Accuracy_Measurements_H,





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



line_plot(df_)


density_plot(df_)



heatmap(df_)

             
    

#####Plotting the scatter plot
scatter_plot(df_,df_.columns[len(df_.columns)-1])
 

########################################################Data Preprocessig Starts from Here
#Now here the datarame has been changed into series type
df_Series_Object=df_.values

scaler=MinMaxScaler(feature_range=(0,1))
scaler=scaler.fit(df_Series_Object)
df_Series_Object=scaler.transform(df_Series_Object)

#divide the data into train and test dataset
train_size=int(len(df_Series_Object)*0.80)
test_size=len(df_Series_Object)-train_size
train,test=df_Series_Object[:train_size,:], df_Series_Object[train_size:len(df_Series_Object ),:]

#index the data into dependent and independent varribles
train_X,train_y=train[:,:-1],train[:,-1]
test_X,test_y=test[:,:-1],test[:,-1]

dict={'Linear Regressor':linear_reg(train_X,train_y,test_X,test_y)[0],'Random Forest Regressor':random_forest(train_X,train_y,test_X,test_y)[0],'Decision Tree Regressor':decision_tree(train_X,train_y,test_X,test_y)[0],'Support Vector Machine Regressor':support_vector_machine(train_X,train_y,test_X,test_y)[0]}

best_model(dict)

_1, test_predict_1,train_predict_1=linear_reg(train_X,train_y,test_X,test_y)

_2, test_predict_2,train_predict_2=random_forest(train_X,train_y,test_X,test_y)

_3, test_predict_3,train_predict_3=decision_tree(train_X,train_y,test_X,test_y)

_4, test_predict_4,train_predict_4=support_vector_machine(train_X,train_y,test_X,test_y)


plt.plot(test_predict_1)

###################Reshaping the train predicted data and test Predicted data#####################
#####Linear Regressor predicted dataset
test_pred_1=test_predict_1.reshape(test_predict_1.shape[0],1)
train_pred_1=train_predict_1.reshape(train_predict_1.shape[0],1)

#####Randdom Forest predicted dataset
test_pred_2=test_predict_2.reshape(test_predict_2.shape[0],1)
train_pred_2=train_predict_2.reshape(train_predict_2.shape[0],1)

#####Lin predicted dataset
test_pred_3=test_predict_3.reshape(test_predict_3.shape[0],1)
train_pred_3=train_predict_3.reshape(train_predict_3.shape[0],1)

#####Linear Regressor predicted dataset
test_pred_4=test_predict_4.reshape(test_predict_4.shape[0],1)
train_pred_4=train_predict_4.reshape(train_predict_4.shape[0],1)


###For Linear Regression Model
All_train_predict_1=concatenate((train_X,train_pred_1),axis=1)
All_test_predict_1=concatenate((test_X,test_pred_1),axis=1)
Concatenated_predict_1=np.concatenate((All_train_predict_1,All_test_predict_1),axis=0)

#For Random forest Model
All_train_predict_2=concatenate((train_X,train_pred_2),axis=1)
All_test_predict_2=concatenate((test_X,test_pred_2),axis=1)
Concatenated_predict_2=np.concatenate((All_train_predict_2,All_test_predict_2),axis=0)

#For Dcisionn Tree Model
All_train_predict_3=concatenate((train_X,train_pred_3),axis=1)
All_test_predict_3=concatenate((test_X,test_pred_3),axis=1)
Concatenated_predict_3=np.concatenate((All_train_predict_3,All_test_predict_3),axis=0)

#For Support vector Machine Model
All_train_predict_4=concatenate((train_X,train_pred_4),axis=1)
All_test_predict_4=concatenate((test_X,test_pred_4),axis=1)
Concatenated_predict_4=np.concatenate((All_train_predict_4,All_test_predict_4),axis=0)

#transforming to original scale of Linear Regression Modell
inv_train_predict_1=scaler.inverse_transform(All_train_predict_1)
inv_test_predict_1=scaler.inverse_transform(All_test_predict_1)
inv_predict_1=scaler.inverse_transform(Concatenated_predict_1)

#transforming to original scale of Linear Regression Modell
inv_train_predict_2=scaler.inverse_transform(All_train_predict_2)
inv_test_predict_2=scaler.inverse_transform(All_test_predict_2)
inv_predict_2=scaler.inverse_transform(Concatenated_predict_2)

#transforming to original scale of Linear Regression Modell
inv_train_predict_3=scaler.inverse_transform(All_train_predict_3)
inv_test_predict_3=scaler.inverse_transform(All_test_predict_3)
inv_predict_3=scaler.inverse_transform(Concatenated_predict_3)

#transforming to original scale of Linear Regression Modell
inv_train_predict_4=scaler.inverse_transform(All_train_predict_4)
inv_test_predict_4=scaler.inverse_transform(All_test_predict_4)
inv_predict_4=scaler.inverse_transform(Concatenated_predict_4)
inv_train_predict_1#transforming to original scale of Linear Regression Modell
inv_train_predict_1=scaler.inverse_transform(All_train_predict_1)
inv_test_predict_1=scaler.inverse_transform(All_test_predict_1)
inv_predict_1=scaler.inverse_transform(Concatenated_predict_1)

#transforming to original scale of Linear Regression Modell
inv_train_predict_2=scaler.inverse_transform(All_train_predict_2)
inv_test_predict_2=scaler.inverse_transform(All_test_predict_2)
inv_predict_2=scaler.inverse_transform(Concatenated_predict_2)

#transforming to original scale of Linear Regression Modell
inv_train_predict_3=scaler.inverse_transform(All_train_predict_3)
inv_test_predict_3=scaler.inverse_transform(All_test_predict_3)
inv_predict_3=scaler.inverse_transform(Concatenated_predict_3)

#transforming to original scale of Linear Regression Modell
inv_train_predict_4=scaler.inverse_transform(All_train_predict_4)
inv_test_predict_4=scaler.inverse_transform(All_test_predict_4)
inv_predict_4=scaler.inverse_transform(Concatenated_predict_4)
inv_train_predict_1

########Extracting the Last 10% values of  Original Data for plotting for clearre view of the prediction################
Df_original=df_lon.iloc[int(len(df_lon)*.9):,-1]
Df_original=pd.DataFrame(Df_original)
Df_original=Df_original.reset_index()
Df_original=Df_original.drop(['index'],axis=1)

##########Extracting the Last 10% values of Predicted Dataset 
Df_predicted_1=inv_predict_1[int(len(df_lon)*.9):,-1]
Df_predicted_1=pd.DataFrame(Df_predicted_1)

##########Extracting the Last 10% values of Predicted Dataset 
Df_predicted_2=inv_predict_2[int(len(df_lon)*.9):,-1]
Df_predicted_2=pd.DataFrame(Df_predicted_2)

##########Extracting the Last 10% values of Predicted Dataset 
Df_predicted_3=inv_predict_3[int(len(df_lon)*.9):,-1]
Df_predicted_3=pd.DataFrame(Df_predicted_3)

##########Extracting the Last 10% values of Predicted Dataset 
Df_predicted_4=inv_predict_4[int(len(df_lon)*.9):,-1]
Df_predicted_4=pd.DataFrame(Df_predicted_4)

line_plot_Preicted_versus_Actual_Dataset(Df_predicted_1, Df_original,"Linear Regressor ")
line_plot_Preicted_versus_Actual_Dataset(Df_predicted_2, Df_original,"Random Forest Regressor ")
line_plot_Preicted_versus_Actual_Dataset(Df_predicted_3, Df_original,"Decision Tree Regressor ")
line_plot_Preicted_versus_Actual_Dataset(Df_predicted_4, Df_original,"Support Vector Machine Regressor ")


Accu_score_model_1=Accuracy_Measurements1(test_y,test_predict_1,"Linear Regressor")
Accu_score_model_2=Accuracy_Measurements1(test_y,test_predict_2,"Random Forest Regressor")
Accu_score_model_3=Accuracy_Measurements1(test_y,test_predict_3,"Decision Tree Regressor")
Accu_score_model_4=Accuracy_Measurements1(test_y,test_predict_4,"Support Vector Machine Regressor")

Accu_score_model_1.update(Accu_score_model_2)
Accu_score_model_1.update(Accu_score_model_3)
Accu_score_model_1.update(Accu_score_model_4)
Accu_score_model_1

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!Depracacted Function !!!!!!!!!!!!!!!!!!!!!!!!
"""
Accuracy_param=("Mean Absolute Error(%)","Mean Squared Error(%)","Root Mean Squared Error(%)","MAPE(%)")
x = np.arange(len(Accuracy_param))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(15)
fig.set_figheight(10)
for attribute, measurement in Accu_score_model_1.items():
#for measurement in penguin_means:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('Accuracy Measurements of All Models')
ax.set_xticks(x + width, Accuracy_param)
ax.legend(loc='upper left', ncols=4)
ax.set_ylim(0, 16)

plt.show()
"""

Accuracy_param=("Mean Absolute Error(%)","Mean Squared Error(%)","Root Mean Squared Error(%)")
x = np.arange(len(Accuracy_param))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(15)
fig.set_figheight(10)
for attribute, measurement in Accu_score_model_1.items():
#for measurement in penguin_means:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('Accuracy Measurements of All Models')
ax.set_xticks(x + width, Accuracy_param)
ax.legend(loc='upper left', ncols=4)
ax.set_ylim(0, 8)

plt.show()



########################################Linear Regression LSTM ,Random Forest_LSTM,Decistion Tree-LSTM &  SVM RRegressor -LSTM Hbrid model strts from here


inv_train_predict_1
inv_test_predict_1
df_Series_Object

#train_y_H_1=df_Series_Object[:train_size,-1]-inv_train_predict_1[:train_size,-1]
#test_y_H_1=df_Series_Object[train_size:,-1]-test_y[:]
train_y_H_1=train_y-train_predict_1
test_y_H_1=test_y-test_predict_1

train_y_H_2=train_y-train_predict_2
test_y_H_2=test_y-test_predict_2

train_y_H_3=train_y-train_predict_3
test_y_H_3=test_y-test_predict_3


#yr=pd.DataFrame(test_y_H_1)


plt.plot(test_y_H_1[930:1010])
#seasonal_decompose(yr, model='additive')

#convert data into sitable dimensin for using it as input in LSTM network
train_X_H_1=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X_H_1=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

train_X_H_2=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X_H_2=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

train_X_H_3=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X_H_3=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

#Hybrid model 1
model_H_1=Sequential()
model_H_1.add(LSTM(64,return_sequences=True,input_shape=(train_X_H_1.shape[1],train_X_H_1.shape[2])))
model_H_1.add(LSTM(32,return_sequences=True))
model_H_1.add(LSTM(16,return_sequences=True))
model_H_1.add(LSTM(8,return_sequences=False))
model_H_1.add(Dropout(0.2))
model_H_1.add(Dense(1))
model_H_1.compile(loss='mae',optimizer='adam',metrics=['mse','accuracy','mae','mape'])
history_H_1=model_H_1.fit(train_X_H_1,train_y_H_1, epochs=30, batch_size=24, validation_data=(test_X_H_1, 
test_y_H_1),verbose=2,shuffle=False)

#Hybrid model 2
model_H_2=Sequential()
model_H_2.add(LSTM(64,return_sequences=True,input_shape=(train_X_H_2.shape[1],train_X_H_2.shape[2])))
model_H_2.add(LSTM(32,return_sequences=True))
model_H_2.add(LSTM(16,return_sequences=True))
model_H_2.add(LSTM(8,return_sequences=False))
model_H_2.add(Dropout(0.2))
model_H_2.add(Dense(1))
model_H_2.compile(loss='mae',optimizer='adam',metrics=['mse','accuracy','mae','mape'])
history_H_2=model_H_2.fit(train_X_H_2,train_y_H_2, epochs=30, batch_size=48, validation_data=(test_X_H_2, 
test_y_H_2),verbose=2,shuffle=False)


#Hybrid Model 3
#Stacked LSTM model 
model_H_3=Sequential()
model_H_3.add(LSTM(64,return_sequences=True,input_shape=(train_X_H_3.shape[1],train_X_H_3.shape[2])))
model_H_3.add(LSTM(32,return_sequences=True))
model_H_3.add(LSTM(16,return_sequences=True))
model_H_3.add(LSTM(8,return_sequences=False))
model_H_3.add(Dropout(0.2))
model_H_3.add(Dense(1))
model_H_3.compile(loss='mae',optimizer='adam',metrics=['mse','accuracy','mae','mape'])
history_H_3=model_H_3.fit(train_X_H_3,train_y_H_3, epochs=30, batch_size=6, validation_data=(test_X_H_3, test_y_H_3),verbose=2,shuffle=False)


model_performace(history_H_1)
model_performace(history_H_2)
model_performace(history_H_3)


#prediction on training and testing dataset for Vanilla Dataset
train_res_predict_H_1=model_H_1.predict(train_X_H_1)
test_res_predict_H_1=model_H_1.predict(test_X_H_1)

#prediction on training and testing dataset for Vanilla Dataset
train_res_predict_H_2=model_H_2.predict(train_X_H_2)
test_res_predict_H_2=model_H_2.predict(test_X_H_2)

#prediction on training and testing dataset for Vanilla Dataset
train_res_predict_H_3=model_H_3.predict(train_X_H_3)
test_res_predict_H_3=model_H_3.predict(test_X_H_3)

######converting from residual to normalized dataset
####for Hybrid Model2
train_predict_H_1=transform_from_res(train_res_predict_H_1,train_predict_1)
test_predict_H_1=transform_from_res(test_res_predict_H_1,test_predict_1)
####for Hybrid Model2
train_predict_H_2=transform_from_res(train_res_predict_H_2,train_predict_2)
test_predict_H_2=transform_from_res(test_res_predict_H_2,test_predict_2)

####for Hybrid Model3
train_predict_H_3=transform_from_res(train_res_predict_H_3,train_predict_3)
test_predict_H_3=transform_from_res(test_res_predict_H_3,test_predict_3)


#converting from three dimension to its original series dataframe
train_X_Original=train_X_H_1.reshape((train_X_H_1.shape[0], train_X_H_1.shape[2]))
test_X_Original=test_X_H_1.reshape((test_X_H_1.shape[0], test_X_H_1.shape[2]))

#For Hybrid Model1
All_train_predict_H_1=concatenate((train_X_Original,train_predict_H_1),axis=1)
All_test_predict_H_1=concatenate((test_X_Original,test_predict_H_1),axis=1)
Concatenated_predict_H_1=np.concatenate((All_train_predict_H_1,All_test_predict_H_1),axis=0)
#For Hybrid Model2
All_train_predict_H_2=concatenate((train_X_Original,train_predict_H_2),axis=1)
All_test_predict_H_2=concatenate((test_X_Original,test_predict_H_2),axis=1)
Concatenated_predict_H_2=np.concatenate((All_train_predict_H_2,All_test_predict_H_2),axis=0)
#For Hybrid Model3
All_train_predict_H_3=concatenate((train_X_Original,train_predict_H_3),axis=1)
All_test_predict_H_3=concatenate((test_X_Original,test_predict_H_3),axis=1)
Concatenated_predict_H_3=np.concatenate((All_train_predict_H_1,All_test_predict_H_3),axis=0)

#transforming to original scale of Hybrid model1
inv_train_predict_H_1=scaler.inverse_transform(All_train_predict_H_1)
inv_test_predict_H_1=scaler.inverse_transform(All_test_predict_H_1)
inv_predict_H_1=scaler.inverse_transform(Concatenated_predict_H_1)
#transforming to original scale of Hybrid Model2
inv_train_predict_H_2=scaler.inverse_transform(All_train_predict_H_2)
inv_test_predict_H_2=scaler.inverse_transform(All_test_predict_H_2)
inv_predict_H_2=scaler.inverse_transform(Concatenated_predict_H_2)
#transforming to original scale of Hybrid model3
inv_train_predict_H_3=scaler.inverse_transform(All_train_predict_H_3)
inv_test_predict_H_3=scaler.inverse_transform(All_test_predict_H_3)
inv_predict_H_3=scaler.inverse_transform(Concatenated_predict_H_3)

#Df_original=df_.iloc[12557:,-1]
########Extracting the Last 10% values of  Original Data for plotting for clearre view of the prediction################
Df_original=df_lon.iloc[int(len(df_lon)*.9):,-1]
Df_original=pd.DataFrame(Df_original)
Df_original=Df_original.reset_index()
Df_original=Df_original.drop(['index'],axis=1)

#For Hybri Model1
Df_predicted_H_1=inv_predict_H_1[int(len(df_lon)*.9):,-1]
#For Hybri Model1
Df_predicted_H_2=inv_predict_H_2[int(len(df_lon)*.9):,-1]
#For Hybri Model3
Df_predicted_H_3=inv_predict_H_3[int(len(df_lon)*.9):,-1]
#ex=Df_predicted_H_3-10

line_plot_Preicted_versus_Actual_Dataset(Df_predicted_H_1,Df_original,"Hybrid Model1")
line_plot_Preicted_versus_Actual_Dataset(Df_predicted_H_2,Df_original,"Hybrid Model2")
line_plot_Preicted_versus_Actual_Dataset(Df_predicted_H_3,Df_original,"Hybrid Model3")

DF_ORIGINAL=df_lon.iloc[train_size:,-1]
DF_ORIGINAL=pd.DataFrame(DF_ORIGINAL)
DF_ORIGINAL=DF_ORIGINAL.reset_index()
DF_ORIGINAL=DF_ORIGINAL.drop(['index'],axis=1)

pre=inv_predict_H_1[train_size:,-1]

Accu_score_test_H_model_1=Accuracy_Measurements_H(test_y_H_1,test_res_predict_H_1,"Hybrid Model 1")
Accu_score_train_H_model_1=Accuracy_Measurements_H(train_y_H_1,train_res_predict_H_1,"Hybrid Model 1")

Accu_score_test_H_model_2=Accuracy_Measurements_H(test_y_H_2,test_res_predict_H_2,"Hybrid Model 2")
Accu_score_train_H_model_2=Accuracy_Measurements_H(train_y_H_2,train_res_predict_H_2,"Hybrid Model 2")

Accu_score_test_H_model_3=Accuracy_Measurements_H(test_y_H_3,test_res_predict_H_3,"Hybrid Model 3")
Accu_score_train_H_model_3=Accuracy_Measurements_H(train_y_H_3,train_res_predict_H_3,"Hybrid Model 3")

######appending the all Dictionary data into one
Accu_score_test_H_model_1.update(Accu_score_test_H_model_2)
Accu_score_test_H_model_1.update(Accu_score_test_H_model_3)
Accu_score_test_H_model_1

################ JUST OSERVING PURPOSE
round(r2_score(test_y_H_1,test_res_predict_H_1) * 100, 2)

Accuracy_param=("Mean Absolute Error(%)","Mean Squared Error(%)","Root Mean Squared Error(%)")
x = np.arange(len(Accuracy_param))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(10)
fig.set_figheight(5)
for attribute, measurement in Accu_score_test_H_model_1.items():
#for measurement in penguin_means:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=1)
    multiplier += 1
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('Accuracy Measurements of All Hybrid Models')
ax.set_xticks(x + width, Accuracy_param)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, .1)

plt.show()



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Deep Learning Models Srarts from Here!!!!!!!!!!!!!!!!!!!!!!!!!!!

#convert data into sitable dimensin for using it as input in LSTM network
test_y_DL=test_y[:]
train_y_DL=train_y[:]
train_X_DL=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X_DL=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
print(train_X_DL.shape,train_y_DL.shape,test_X_DL.shape,test_y_DL.shape)


#Simple Forward LSTM model Single directional
model_1=Sequential()
model_1.add(LSTM(40,return_sequences=False,input_shape=(train_X_DL.shape[1],train_X_DL.shape[2])))
model_1.add(Dropout(0.2))
model_1.add(Dense(1))
model_1.compile(loss='mae',optimizer='adam',metrics=['mse','accuracy','mae','mape'])
history_1=model_1.fit(train_X_DL,train_y_DL, epochs=25, batch_size=24, validation_data=(test_X_DL, test_y_DL),verbose=2,shuffle=False)


#Bidirectional LSTM model 
model_2=Sequential()
model_2.add(Bidirectional(LSTM(40,return_sequences=True,input_shape=(train_X_DL.shape[1],train_X_DL.shape[2]))))
model_2.add(Bidirectional(LSTM(20,return_sequences=False)))
model_2.add(Dropout(0.2))
model_2.add(Dense(1))
model_2.compile(loss='mae',optimizer='adam',metrics=['mse','accuracy','mae','mape'])
history_2=model_2.fit(train_X_DL,train_y_DL, epochs=25, batch_size=24, validation_data=(test_X_DL, test_y_DL),verbose=2,shuffle=False)


#Stacked LSTM model 
model_3=Sequential()
model_3.add(LSTM(64,return_sequences=True,input_shape=(train_X_DL.shape[1],train_X_DL.shape[2])))
model_3.add(LSTM(32,return_sequences=True))
model_3.add(LSTM(16,return_sequences=True))
model_3.add(LSTM(8,return_sequences=False))
model_3.add(Dropout(0.2))
model_3.add(Dense(1))
model_3.compile(loss='mae',optimizer='adam',metrics=['mse','accuracy','mae','mape'])
history_3=model_3.fit(train_X_DL,train_y_DL, epochs=25, batch_size=96, validation_data=(test_X_DL, test_y_DL),verbose=2,shuffle=False)

model_performace(history_1)
model_performace(history_2)
model_performace(history_3)
 
 
#prediction on training and testing dataset for Vanilla Dataset
train_predict_1_DL=model_1.predict(train_X_DL)
test_predict_1_DL=model_1.predict(test_X_DL)

#prediction on training and testing dataset for Vanilla Dataset
train_predict_2_DL=model_2.predict(train_X_DL)
test_predict_2_DL=model_2.predict(test_X_DL)

#prediction on training and testing dataset for Vanilla Dataset
train_predict_3_DL=model_3.predict(train_X_DL)
test_predict_3_DL=model_3.predict(test_X_DL)


#converting from three dimension to its original series dataframe
train_X_Original_DL=train_X_DL.reshape((train_X_DL.shape[0], train_X_DL.shape[2]))
test_X_Original_DL=test_X_DL.reshape((test_X_DL.shape[0], test_X_DL.shape[2]))

#For vanilla LSTM model
All_train_predict_1_DL=concatenate((train_X_Original_DL,train_predict_1_DL),axis=1)
All_test_predict_1_DL=concatenate((test_X_Original_DL,test_predict_1_DL),axis=1)
Concatenated_predict_1_DL=np.concatenate((All_train_predict_1_DL,All_test_predict_1_DL),axis=0)                             

#For vanilla LSTM model
All_train_predict_2_DL=concatenate((train_X_Original_DL,train_predict_2_DL),axis=1)
All_test_predict_2_DL=concatenate((test_X_Original_DL,test_predict_2_DL),axis=1)
Concatenated_predict_2_DL=np.concatenate((All_train_predict_2_DL,All_test_predict_2_DL),axis=0)                             

#For stacked LSTM model
All_train_predict_3_DL=concatenate((train_X_Original_DL,train_predict_3_DL),axis=1)
All_test_predict_3_DL=concatenate((test_X_Original_DL,test_predict_3_DL),axis=1)
Concatenated_predict_3_DL=np.concatenate((All_train_predict_3_DL,All_test_predict_3_DL),axis=0) 
train_X_DL.shape


#transforming to original scale of Vanilla LSTM
inv_train_predict_1_DL=scaler.inverse_transform(All_train_predict_1_DL)
inv_test_predict_1_DL=scaler.inverse_transform(All_test_predict_1_DL)
inv_predict_1_DL=scaler.inverse_transform(Concatenated_predict_1_DL)

#transforming to original scale of Bidirectional LSTM
inv_train_predict_2_DL=scaler.inverse_transform(All_train_predict_2_DL)
inv_test_predict_2_DL=scaler.inverse_transform(All_test_predict_2_DL)
inv_predict_2_DL=scaler.inverse_transform(Concatenated_predict_2_DL)

#transforming to original scale of stacked LSTM
inv_train_predict_3_DL=scaler.inverse_transform(All_train_predict_3_DL)
inv_test_predict_3_DL=scaler.inverse_transform(All_test_predict_3_DL)
inv_predict_3_DL=scaler.inverse_transform(Concatenated_predict_3_DL)

Df_original_DL=df_lon.iloc[train_size:,-1]
df_series_DL=Df_original_DL.values
Df_original_DL=pd.DataFrame(df_series_DL)

Df_predicted_1_DL=inv_predict_1_DL[train_size:,-1]
Df_predicted_2_DL=inv_predict_2_DL[train_size:,-1]
Df_predicted_3_DL=inv_predict_3_DL[train_size:,-1]

line_plot_Preicted_versus_Actual_Dataset(Df_predicted_1_DL,Df_original_DL,"ML")
line_plot_Preicted_versus_Actual_Dataset(Df_predicted_2_DL,Df_original_DL,"ML")
line_plot_Preicted_versus_Actual_Dataset(Df_predicted_3_DL,Df_original_DL,"ML")

Accu_score_test_DL_model_1=Accuracy_Measurements_DL(test_y_DL,test_predict_1_DL,"ML Model 1")
Accu_score_test_DL_model_2=Accuracy_Measurements_DL(test_y_DL,test_predict_2_DL,"ML Model 2")
Accu_score_test_DL_model_3=Accuracy_Measurements_DL(test_y_DL,test_predict_3_DL,"ML Model 3")
######appending the all Dictionary data into one
Accu_score_test_DL_model_1.update(Accu_score_test_DL_model_2)
Accu_score_test_DL_model_1.update(Accu_score_test_DL_model_3)
Accu_score_test_DL_model_1

########JUST FOR OSERVATION
round(r2_score(test_y_DL,test_predict_3_DL) * 100, 2)

Accuracy_param=("Mean Absolute Error(%)","Mean Squared Error(%)","Root Mean Squared Error(%)")
x = np.arange(len(Accuracy_param))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(10)
fig.set_figheight(5)
for attribute, measurement in Accu_score_test_DL_model_1.items():
#for measurement in penguin_means:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=1)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('Accuracy Measurements of All Hybrid Models')
ax.set_xticks(x + width, Accuracy_param)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)

plt.show()


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!CNN-LSTM Hybrid model starts from here!!!!!!!!!!!!!!!!


df_H=pd.DataFrame(df_Series_Object)

X_H,y_H=df_X_y(df_H,192)



train_size=int(len(df_H)*.8)
train_X_H,train_y_H=X_H[:train_size],y_H[:train_size]
test_X_H,test_y_H=X_H[train_size:],y_H[train_size:]

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Depracated Version!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
model_H = Sequential()
model_H.add(Conv1D(200, kernel_size=3, activation='relu', input_shape=(192, 18)))
model_H.add(LSTM(200))
model_H.add(Dense(100))
model_H.add(Dense(1, activation='relu'))
model_H.summary()
"""

model_H = Sequential()
model_H.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(192, 18)))
model_H.add(LSTM(64))
model_H.add(Flatten())
model_H.add(Dense(8, activation='relu'))
model_H.add(Dense(2,'linear'))
model_H.summary()

model_H.compile(loss='mae',optimizer='adam',metrics=['mse','accuracy','mae','mape'])
history_H=model_H.fit(train_X_H,train_y_H, epochs=25, batch_size=48, validation_data=(test_X_H, 
test_y_H),verbose=2,shuffle=False)

model_performace(history_H)

test_predict_twoD_H=model_H.predict(test_X_H)
train_predict_twoD_H_=model_H.predict(train_X_H)

test_predict_H=twoD_oneD(test_predict_twoD_H)
train_predict_H=twoD_oneD(train_predict_twoD_H_)


test_predict_H.shape

plt.plot(test_predict_H)
plt.plot(test_y)


#converting from three dimension to its original series dataframe
train_X_Original_H=pd.DataFrame(transform_(train_X_H))
test_X_Original_H=pd.DataFrame(transform_(test_X_H))

All_train_predict_H=concatenate((train_X_Original_H,train_predict_H),axis=1)
All_test_predict_H=concatenate((test_X_Original_H,test_predict_H),axis=1)
Concatenated_predict_H=np.concatenate((All_train_predict_H,All_test_predict_H),axis=0) 


######!!!!!!!WARNING : Contains drop method
Concatenated_predict_H=pd.DataFrame(Concatenated_predict_H)
Concatenated_predict_H=Concatenated_predict_H.drop([len(Concatenated_predict_H.columns)-2],axis=1)

All_train_predict_H=pd.DataFrame(All_train_predict_H)
All_train_predict_H=All_train_predict_H.drop([len(All_train_predict_H.columns)-2],axis=1)

All_test_predict_H=pd.DataFrame(All_test_predict_H)
All_test_predict_H=All_test_predict_H.drop([len(All_test_predict_H.columns)-2],axis=1)

#transforming to original scale of Vanilla LSTM
inv_predict_H=scaler.inverse_transform(Concatenated_predict_H)
inv_train_predict_H=scaler.inverse_transform(All_train_predict_H)
inv_test_predict_H=scaler.inverse_transform(All_test_predict_H)

inv_predicted_H=pd.DataFrame(inv_predict_H)
inv_train_predict_H=pd.DataFrame(All_train_predict_H)
inv_test_predict_H=pd.DataFrame(All_test_predict_H)

plt.plot(inv_predicted_H[17][train_size:])
plt.plot(df_lon.iloc[train_size:,-1])
inv_predict_H.shape


Accu_score_test_model_H=Accuracy_Measurements_H(test_y_H,test_predict_H,"Hybrid Model 1")

Accu_score_test_model_H


########JUST FOR OBSERVATIONAL PURPOSE
round(r2_score(test_y_H,test_predict_H) * 100, 2)


Accuracy_param=("Mean Absolute Error(%)","Mean Squared Error(%)","Root Mean Squared Error(%)")
x = np.arange(len(Accuracy_param))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')
fig.set_figwidth(8)
fig.set_figheight(5)
for attribute, measurement in Accu_score_test_model_H.items():
#for measurement in penguin_means:
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute,color=['orange','orange','orange'])
    ax.bar_label(rects, padding=1)
    multiplier += 1
ax.set_ylabel('Error')
ax.set_title('Accuracy Measurements of Hybrid Model')
ax.set_xticks(x + width, Accuracy_param)
ax.legend(loc='upper left', ncols=4)
ax.set_ylim(0, 5)

plt.show()