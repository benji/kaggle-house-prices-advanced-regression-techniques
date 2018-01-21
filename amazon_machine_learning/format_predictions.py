# Input from AML:
#tag,trueLabel,score
#1461,,1.225708E5
#1462,,1.516432E5

# Desired Output
#Id,SalePrice
#1461,169277.0524984

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import math

default_SalePrice = 100000

# load data
df = pd.read_csv('data/bp-test14-5QMXGM4XHOWVEKEI-test_transformed.csv')

df = df[['tag', 'score']]
df.columns = ['Id', 'SalePrice']

#df['SalePrice']=df['SalePrice'].apply(lambda x:math.exp(x))

df_neg = df[df['SalePrice'] < 0]
print 'Replacing ' + str(df_neg.shape[0]) + ' negative values.'
df['SalePrice'][df['SalePrice'] < 0] = default_SalePrice

missing = 0
for id in range(1461, 2920):
    if df[df['Id'] == id].size == 0:
        missing = missing + 1
        df.loc[len(df)] = [id, default_SalePrice]

print 'Assigned default values to ' + str(missing) + ' missing rows.'

# save
df.to_csv('data/predictions_formatted.csv', sep=',', index=False)