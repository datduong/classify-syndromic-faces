
import os,sys,re,pickle
from turtle import shape
import numpy as np 
import pandas as pd 

fout = '/data/duongdb/ManyFaceConditions01312022/pretrain-50knormal-gender.csv'

df = '/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-BlankBackgroundCenter-label_formated.csv' 
df2 = '/data/duongdb/WS22qOther_08102021/Classify/train+blankcenter+WS+22q11DS+Control+Normal+Whole.csv'

df = pd.read_csv(df) # need, name,path,label,fold,is_ext
temp_ = [re.sub('_face0','',f.split('/')[-1]) for f in df['name'].values]
df['name'] = temp_

df2 = pd.read_csv(df2) # need, name,path,label,fold,is_ext
df2 = df2[df2['label'].str.contains('Normal')]

# set ( df2['name'].values ) - set(df['name'].values )

df = pd.merge(df2,df,on='name')

# predict gender as pre-train 
label = []
for g in df['Female'].values: 
  g = np.round(g,decimals=0)
  if g==1: label.append('female')
  else: label.append('male')

#
df['label'] = label 

df['is_ext'] = 0 

# 

df.to_csv(fout,index=False)
print (df.shape)
