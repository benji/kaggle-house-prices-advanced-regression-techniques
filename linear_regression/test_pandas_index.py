import pandas as pd
import numpy as np
import sys

df = pd.DataFrame(columns=['a'],data=['a1','a2','a3','a4'])

print df
print 'loc[2]=',df.loc[2].values

print 'deleting idx 1'
df.drop(1,inplace=True)

print df
print 'loc[2]=',df.loc[2].values