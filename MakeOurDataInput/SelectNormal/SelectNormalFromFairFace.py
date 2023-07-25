
import os,sys,re,pickle 
import numpy as np 
import pandas as pd 

# ! pick the same 10k normal? 

NUM_PICK = 600 
NUM_TEST = 50

mainpath = '/data/duongdb/ManyFaceConditions01312022/'
foutimg = os.path.join(mainpath,'Normal'+str(NUM_PICK)+'SubsetTrainTest')

df = pd.read_csv(os.path.join(mainpath,'ManyCondition+10kNormal-Other.csv'))

# df['labels'].value_counts()

# Normalyoungadult           2422
# Normalolderadult           2369
# Normalyoungchild           2166
# Normaladolescence          1993
# Normal2y                    709

normal_label = ['Normalyoungadult','Normalolderadult','Normalyoungchild','Normaladolescence','Normal2y']

df = df [ df['label'].isin(normal_label) ] 

newdf = []
for i,l in enumerate(normal_label): 
  temp = df[df['label'] == l]
  temp = temp.sample(n=NUM_PICK,random_state=i).reset_index(drop=True)
  newdf.append(temp)

# ---------------------------------------------------------------------------- #

newdf = pd.concat(newdf)
newdf = newdf.reset_index(drop=True)
newdf = newdf.sample(frac=1,random_state=2022).reset_index(drop=True)

fold = [i%5 for i in range(newdf.shape[0])] 
newdf ['fold'] = fold 

newdf.to_csv(os.path.join(mainpath,'Train-'+str(NUM_PICK)+'EachNormal.csv'),index=None)

# ---------------------------------------------------------------------------- #
# ! copy ? 
if not os.path.exists(foutimg): 
  os.mkdir(foutimg)

#
for im in newdf['path'].values: 
  os.system ('scp ' + im + ' ' + foutimg)

# ---------------------------------------------------------------------------- #
# ! test set, so need 6 folds 

dftest = df [ ~df['name'].isin(newdf['name'].values) ] 

newdf = []
for i,l in enumerate(normal_label): 
  temp = dftest[dftest['label'] == l]
  temp = temp.sample(n=NUM_TEST,random_state=i).reset_index(drop=True)
  newdf.append(temp)

# 
newdf = pd.concat(newdf)
newdf = newdf.reset_index(drop=True)

newdf ['fold'] = 5 # ! test set
newdf.to_csv(os.path.join(mainpath,'Test-'+str(NUM_TEST)+'EachNormal.csv'),index=None)

# ! external data
df['is_ext'] = 1


for im in newdf['path'].values: 
  os.system ('scp ' + im + ' ' + foutimg)


