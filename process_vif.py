# Importing the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import csv
import sys

from sklearn.datasets import load_boston

# Initializing the data
csv_file = open('dsaa_dataset_order_rename.csv')
rows = csv.reader(csv_file)
rows = list(rows)
data = pd.DataFrame(rows[1:]).astype(float)

# __console = sys.stdout
# log_location = 'all_in.log'
# log = open(log_location,'a+')
# sys.stdout = log

# preprocessing and check the data
print(data.shape)
data.columns = rows[0]
data = data.drop('id', axis=1)
data = data.drop('TtlPrc', axis=1)
data = data.rename(columns={"UntPrc": "Price"})

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
print(data.describe(include='all'))

# Finding out the correlation between the features
corr = data.corr()
print(corr.shape)

# Plotting the heatmap of correlation between features
# plt.figure(figsize=(20,20))
# sns.set(font_scale=1.5)
# sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')
# plt.show()

# data_scale = data.max()-data.min()
# data_min = data.min()
#
# normalized_data=(data-data_min)/(data_scale)
# data = normalized_data

# Spliting target variable and independent variables
X = data.drop(['Price'], axis = 1)
y = data['Price']

# load statmodels functions
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


X = add_constant(X)

results = pd.Series([variance_inflation_factor(X.values, i)
               for i in range(X.shape[1])],
              index=X.columns)

# print(results.shape)
# results['index'] = results.index

# print(results.head())

print(results.sort_values(axis=0, ascending=False))
