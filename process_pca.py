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

data_scale = data.max()-data.min()
data_min = data.min()

normalized_data=(data-data_min)/(data_scale)
data = normalized_data

# Spliting target variable and independent variables
X = data.drop(['Price'], axis = 1)
y = data['Price']

import pandas
from sklearn.decomposition import PCA
import numpy
import matplotlib.pyplot as plot

df = data

# You must normalize the data before applying the fit method
# df_normalized=(df - df.mean()) / df.std()
pca = PCA(n_components=df.shape[1])
pca.fit(data)

# Reformat and view results
loadings = pandas.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(data.columns))],
index=df.columns)
print(loadings)

plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')
plot.show()