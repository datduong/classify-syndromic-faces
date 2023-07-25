
import os,sys,re,pickle 
import numpy as np
import pandas as pd 

# ! rename the path. 

newpath = '/data/duongdb/ManyFaceConditions04202022/Normal500SubsetTrainTest-align255pix-nobg'
newdfpath = '/data/duongdb/ManyFaceConditions04202022/Train-500EachNormal-align255pix-nobg.csv' # ! new output 

oldpath = '/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-BlankBackgroundCenter'
olddfpath = '/data/duongdb/ManyFaceConditions01312022/Train-600EachNormal.csv' # ! were used in prelim experiments

df = pd.read_csv(olddfpath,dtype=str)

temp = [re.sub (oldpath, newpath, i) for i in df['path'].values ]
df['path'] = temp 

# df[0:10]

# ! keep just images in @newpath
images_to_keep = os.listdir(newpath)

df = df[df['name'].isin(images_to_keep)].reset_index(drop=True)
print (df)

# ---------------------------------------------------------------------------- #
# ! check file exists. 
notexist = []
exist = []
for i in df['path'].values: 
  if not os.path.exists(i): 
    notexist.append (i)
  else: 
    exist.append(i)

# 
print(len(notexist))



# ---------------------------------------------------------------------------- #

df['detail_label'] = df['label']
df['label'] = 'Normal'

df.to_csv(newdfpath,index=None)


