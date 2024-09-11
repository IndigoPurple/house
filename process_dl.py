# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import csv

from sklearn.datasets import load_boston

# Initializing the data
csv_file = open('dsaa_dataset_order_rename.csv')
rows = csv.reader(csv_file)
rows = list(rows)
data = pd.DataFrame(rows[1:]).astype(float)

# preprocessing and check the data
print(data.shape)
data.columns = rows[0]
data = data.drop('id', axis=1)
data = data.drop('TtlPrc', axis=1)
data = data.rename(columns={"UntPrc": "Price"})

pd.set_option('display.max_columns', None)

use_property_only = 0
remove_a, remove_t, remove_s = 0,0,0
if use_property_only:
    data = data.drop(['TspNum','TspDst', 'AtrNum','AtrDst','EdcNum','EdcDst','HthNum',
                      'HthDst','RstNum','RstDst','RtlNum','RtlDst','TrfV',
                      'AgrPct','DstPct','HppPct','SadPct','FeaPct'], axis=1)
else:
    if remove_a:
        data = data.drop(['TspNum', 'TspDst', 'AtrNum', 'AtrDst', 'EdcNum', 'EdcDst', 'HthNum',
                          'HthDst', 'RstNum', 'RstDst', 'RtlNum', 'RtlDst'], axis=1)
    if remove_t:
        data = data.drop(['TrfV'], axis=1)
    if remove_s:
        data = data.drop(['AgrPct', 'DstPct', 'HppPct', 'SadPct', 'FeaPct'], axis=1)


# print(data.head())
# print(data.dtypes)
# print(data.isnull().sum())

# Viewing the data statistics
# print(data.describe().T)

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# ######### remove amenityNum = 100
# data = data[~(data['TspNum'] == 100)]
# data = data[~(data['AtrNum'] == 100)]
# data = data[~(data['EdcNum'] == 100)]
# data = data[~(data['HthNum'] == 100)]
# data = data[~(data['RstNum'] == 100)]
# data = data[~(data['RtlNum'] == 100)]
# data = data[~(data['TspDst'] == 1000)]
# data = data[~(data['AtrDst'] == 1000)]
# data = data[~(data['EdcDst'] == 1000)]
# data = data[~(data['HthDst'] == 1000)]
# data = data[~(data['RstDst'] == 1000)]
# data = data[~(data['RtlDst'] == 1000)]
# print(np.shape(data))
#
# fig, axs = plt.subplots(ncols=9, nrows=3, figsize=(20, 10))
# index = 0
# axs = axs.flatten()
# for k,v in data.items():
#     sns.boxplot(y=k, data=data, ax=axs[index])
#     index += 1
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# plt.show()
#
# # for k, v in data.items():
# #     q1 = v.quantile(0.25)
# #     q3 = v.quantile(0.75)
# #     irq = q3 - q1
# #     v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
# #     perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
# #     print("Column %s outliers = %.2f%%" % (k, perc))
#
# plt.clf()
# fig, axs = plt.subplots(ncols=9, nrows=3, figsize=(20, 10))
# index = 0
# axs = axs.flatten()
# for k,v in data.items():
#     sns.distplot(v, ax=axs[index])
#     index += 1
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# plt.show()
# exit()

# plt.clf()
# plt.figure(figsize=(20, 10))
# sns.heatmap(data.corr().abs(),  annot=True)
# plt.show()

# from sklearn import preprocessing
# # Let's scale the columns before plotting them against MEDV
# min_max_scaler = preprocessing.MinMaxScaler()
# # column_sels = ['AtrNum', 'EdcNum', 'HthNum', 'RstNum', 'TrfV', 'DstPct']
# column_sels = ['Lat', 'Lng', 'Year']
# x = data.loc[:,column_sels]
# y = data['Price']
# x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
# fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
# index = 0
# axs = axs.flatten()
# for i, k in enumerate(column_sels):
#     sns.regplot(y=y, x=x[k], ax=axs[i], line_kws={"color": "red"})
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# plt.show()
#
# exit()

# Finding out the correlation between the features
# corr = data.corr()
# print(corr.shape)

# Plotting the heatmap of correlation between features
# plt.figure(figsize=(20,20))
# sns.set(font_scale=1.5)
# sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
# plt.show()

data_scale = data.max()-data.min()
data_min = data.min()

normalized_data=(data-data_min)/(data_scale)
data = normalized_data

# Spliting target variable and independent variables
X = data.drop(['Price'], axis = 1)
y = data['Price']

# normalized_X=(X-X.min())/(X.max()-X.min())
# X = normalized_X

# Splitting to training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)

from tensorflow.keras import Sequential    # import Sequential from tensorflow.keras
from tensorflow.keras.layers import Dense  # import Dense from tensorflow.keras.layers
from numpy.random import seed     # seed helps you to fix the randomness in the neural network.
import tensorflow

# import RMSprop optimizer
from tensorflow.keras.optimizers import RMSprop,Adam

####################### Complete example to check the performance of the model with different learning rates #######################################
n_features = X.shape[1]
# define the model
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(n_features,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
# model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(8192, activation='relu'))
# model.add(Dense(4096, activation='relu'))
# model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

optimizer = Adam(0.0001)    # 0.1 is the learning rate
model.compile(loss='mean_squared_error',optimizer=optimizer)    # compile the model

# fit the model
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose = 1)

# y_test_pred = model.predict(X_test)
y_test_pred = model.predict(X_test) * data_scale['Price'] + data_min['Price']
y_test = y_test * data_scale['Price'] + data_min['Price']

# Model Evaluation
acc_linreg = metrics.r2_score(y_test, y_test_pred)
mae_linreg = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_linreg = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_linreg)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_linreg)

# evaluate the model
# print('The MSE value is: ', model.evaluate(X_test, y_test) * data_scale + data_min)

# # Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_test, y_test_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices", labelpad=1.5)
# plt.title("Prices vs Predicted prices")
# xpoints = ypoints = plt.xlim()
# plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=3, scalex=False, scaley=False)
# plt.show()
#
# # Checking residuals
# plt.clf()
# plt.scatter(y_test_pred,y_test-y_test_pred[0])
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals", labelpad=1.5)
# plt.show()
#
# # Checking Normality of errors
# plt.clf()
# sns.distplot(y_test-y_test_pred[0])
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

exit()

print('------------Linear Regression-----------------')
# Import library for Linear Regression
from sklearn.linear_model import LinearRegression

# Create a Linear regressor
lm = LinearRegression()

# Train the model using the training sets
lm.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)
#
# # Value of y intercept
# print(lm.intercept_)
#
# #Converting the coefficient values to a dataframe
# coeffcients = pd.DataFrame([X_train.columns,lm.coef_]).T
# coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})
# print(coeffcients)
#
# # Model prediction on train data
y_pred = lm.predict(X_train)
# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_train, y_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices")
# plt.title("Prices vs Predicted prices")
# plt.show()

# Checking residuals
# plt.clf()
# plt.scatter(y_pred,y_train-y_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals")
# plt.show()

# Checking Normality of errors
# plt.clf()
# sns.distplot(y_train-y_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

# Predicting Test data with the model
y_test_pred = lm.predict(X_test)
# Model Evaluation
acc_linreg = metrics.r2_score(y_test, y_test_pred)
mae_linreg = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_linreg = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_linreg)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_linreg)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_linreg)

# # Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_test, y_test_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices", labelpad=1.5)
# plt.title("Prices vs Predicted prices")
# xpoints = ypoints = plt.xlim()
# plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=3, scalex=False, scaley=False)
# plt.show()
#
# # Checking residuals
# plt.clf()
# plt.scatter(y_test_pred,y_test-y_test_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals", labelpad=1.5)
# plt.show()
#
# # Checking Normality of errors
# plt.clf()
# sns.distplot(y_test-y_test_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

print('------------Random Forest Regressor-----------------')
# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg = RandomForestRegressor()

# Train the model using the training sets
reg.fit(X_train, y_train)

# Model prediction on train data
y_pred = reg.predict(X_train)

# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_rf = metrics.r2_score(y_test, y_test_pred)
mae_rf = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_rf = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_rf)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_rf)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_rf)

# # Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_test, y_test_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices", labelpad=1.5)
# plt.title("Prices vs predicted Prices")
# xpoints = ypoints = plt.xlim()
# plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=3, scalex=False, scaley=False)
#
# plt.show()
#
# # Checking residuals
# plt.scatter(y_test_pred,y_test-y_test_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals", labelpad=1.5)
# plt.show()
#
# # Checking Normality of errors
# plt.clf()
# sns.distplot(y_test-y_test_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency", labelpad=1.5)
# plt.show()

# fig,ax = plt.subplots(figsize=(10,10))
# global_importances = pd.Series(reg.feature_importances_, index=X_train.columns)
# global_importances.sort_values(ascending=True, inplace=True)
# global_importances.plot.barh(color='green',grid=True)
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.title("Feature Importance")
# plt.show()
# exit()

# print('------------XGBoost Regressor-----------------')
# Import XGBoost Regressor
from xgboost import XGBRegressor

#Create a XGBoost Regressor
reg = XGBRegressor()

# Train the model using the training sets
reg.fit(X_train, y_train)

# Model prediction on train data
y_pred = reg.predict(X_train)

# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))

# Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_train, y_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices")
# plt.title("Prices vs Predicted prices")
# plt.show()

# Checking residuals
# plt.scatter(y_pred,y_train-y_pred)
# plt.title("Predicted vs residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals")
# plt.show()

# Checking Normality of errors
# plt.clf()
# sns.distplot(y_train-y_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency")
# plt.show()

#Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_xgb = metrics.r2_score(y_test, y_test_pred)
mae_xgb = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_xgb = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_xgb)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_xgb)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_xgb)

# Visualizing the differences between actual prices and predicted values
# plt.clf()
# plt.scatter(y_test, y_test_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices", labelpad=1.5)
# plt.title("Prices vs Predicted Prices")
# xpoints = ypoints = plt.xlim()
# plt.plot(xpoints, ypoints, linestyle='-', color='r', lw=3, scalex=False, scaley=False)
#
# plt.show()
#
# # Checking residuals
# plt.scatter(y_test_pred,y_test-y_test_pred)
# plt.title("Predicted vs Residuals")
# plt.xlabel("Predicted")
# plt.ylabel("Residuals", labelpad=1.5)
# plt.show()
#
# # Checking Normality of errors
# plt.clf()
# sns.distplot(y_test-y_test_pred)
# plt.title("Histogram of Residuals")
# plt.xlabel("Residuals")
# plt.ylabel("Frequency", labelpad=1.5)
# plt.show()

from xgboost import plot_importance
# plt.clf()
# fig,ax = plt.subplots(figsize=(10,10))
# plot_importance(reg,height=0.5,max_num_features=64,ax=ax)
# plt.show()

print('------------SVM Regressor-----------------')
# Creating scaled set to be used in model to improve our results
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Import SVM Regressor
from sklearn import svm

# Create a SVM Regressor
reg = svm.SVR()
# Train the model using the training sets
reg.fit(X_train, y_train)

# Model prediction on train data
y_pred = reg.predict(X_train)

# Model Evaluation
print('R^2:',metrics.r2_score(y_train, y_pred))
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(y_train, y_pred))
print('MSE:',metrics.mean_squared_error(y_train, y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
#
# # Visualizing the differences between actual prices and predicted values
# # plt.clf()
# # plt.scatter(y_train, y_pred)
# # plt.xlabel("Prices")
# # plt.ylabel("Predicted prices")
# # plt.title("Prices vs Predicted prices")
# # plt.show()
#
# # Checking residuals
# # plt.scatter(y_pred,y_train-y_pred)
# # plt.title("Predicted vs residuals")
# # plt.xlabel("Predicted")
# # plt.ylabel("Residuals")
# # plt.show()
#
# # Checking Normality of errors
# # plt.clf()
# # sns.distplot(y_train-y_pred)
# # plt.title("Histogram of Residuals")
# # plt.xlabel("Residuals")
# # plt.ylabel("Frequency")
# # plt.show()
#
# Predicting Test data with the model
y_test_pred = reg.predict(X_test)

# Model Evaluation
acc_svm = metrics.r2_score(y_test, y_test_pred)
mae_svm = metrics.mean_absolute_error(y_test, y_test_pred)
rmse_svm = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('R^2:', acc_svm)
print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print('MAE:',mae_svm)
print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))
print('RMSE:',rmse_svm)

# print('------------Evaluation and comparision of all the models-----------------')
# models = pd.DataFrame({
#     'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines'],
#     'R-squared Score': [acc_linreg*100, acc_rf*100, acc_xgb*100, acc_svm*100],
#     'MAE': [mae_linreg, mae_rf, mae_xgb, mae_svm],
#     'RMSE': [rmse_linreg, rmse_rf, rmse_xgb, rmse_svm]})
# print(models.sort_values(by='R-squared Score', ascending=True))
# print(models.sort_values(by='MAE', ascending=True))
# print(models.sort_values(by='RMSE', ascending=True))