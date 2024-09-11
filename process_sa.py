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
print(data.columns.values)
# exit()
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


print(data.head())
print(data.dtypes)
print(data.isnull().sum())

# Viewing the data statistics
print(data.describe().T)


# Finding out the correlation between the features
corr = data.corr()
print(corr.shape)

# Plotting the heatmap of correlation between features
# plt.figure(figsize=(20,20))
# sns.set(font_scale=1.5)
# sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
# plt.show()

# Spliting target variable and independent variables
X = data.drop(['Price'], axis = 1)
y = data['Price']
# Splitting to training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)

from sklearn.datasets import make_regression
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

class Simulate:
    def __init__(self, obs, var):
        self.obs = obs
        self.var = var

    def simulate_increase(self, model, percentage):
        baseline = model.predict(self.obs)
        plus = {}
        for ivar in self.var:
            X_plus = self.obs.copy()
            X_plus[ivar] = X_plus[ivar] - X_plus[ivar] * (percentage / 100)
            plus[ivar] = model.predict(X_plus)
        b = pd.DataFrame(
            plus, index=['simulated'
                         ]).T.reset_index().rename(columns={'index': 'test'})
        b['baseline'] = baseline[0]
        return b

    @staticmethod
    def plot_simulation(d, **kwargs):
        fig, ax = plt.subplots()
        sns.barplot(x='test', y='simulated', data=d, palette='deep', ax=ax)
        ax.axhline(d['baseline'].values[0], color='grey', linestyle='--', linewidth=2)
        ax.plot([0, 0], [-100, -100], color='grey', linestyle='--', linewidth=2, label='baseline')

        maxi = int(d['simulated'].max() + d['simulated'].max() * 0.1)
        mini = int(d['simulated'].min() - d['simulated'].min() * 0.1)
        ax.set_ylim([mini, maxi])

        ax.set_xlabel('Simulated variables')
        ax.set_ylabel('Target value')
        ax.set_title(kwargs.get('title'))
        ax.legend()

        ax.grid(axis='y', linewidth=.3)
        sns.despine(offset=10, trim=True)
        plt.tight_layout()
        plt.show()


# VAR_OPTIMIZE = data.columns.values[1::]
# VAR_OPTIMIZE = ['Year', 'Elvt', 'RmNum', 'HllNum', 'KchNum', 'BthNum',
#  'Lat', 'Lng', 'TspNum', 'TspDst', 'AtrNum', 'AtrDst', 'EdcNum', 'EdcDst',
#  'HthNum', 'HthDst', 'RstNum', 'RstDst', 'RtlNum', 'RtlDst', 'TrfV', 'AgrPct',
 # 'DstPct', 'HppPct', 'SadPct', 'FeaPct']
VAR_OPTIMIZE = ['Year', 'Elvt', 'RmNum', 'HllNum', 'KchNum', 'BthNum',
                'Lat', 'Lng']
# VAR_OPTIMIZE = ['TspNum', 'AtrNum', 'EdcNum','HthNum', 'RstNum',  'RtlNum']
# VAR_OPTIMIZE = ['TspDst', 'AtrDst', 'EdcDst', 'HthDst', 'RstDst', 'RtlDst']
# VAR_OPTIMIZE = ['TrfV', 'AgrPct','DstPct', 'HppPct', 'SadPct', 'FeaPct']
PERC = 5
ROW = X.iloc[[29]]

S = Simulate(obs=ROW, var=VAR_OPTIMIZE)
d = S.simulate_increase(model=model, percentage=PERC)
# S.plot_simulation(d, title=f'Impact of a {PERC}% increase of {VAR_OPTIMIZE} in target value')
S.plot_simulation(d, title=f'Impact of a {PERC}% decrease of property variable value')
exit()

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