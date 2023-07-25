import os,sys,re,pickle 
import numpy as np 
import pandas as pd
from pip import main 

# ---------------------------------------------------------------------------- #
# ! some rm background and align may not work, need to retain "good" images 
mainpath = '/data/duongdb/ManyFaceConditions01312022/'
headername = 'Train'
NUM_PICK = 500
df = os.path.join(mainpath,headername+'-600EachNormal.csv')
img_path = os.path.join(mainpath,'Normal600SubsetTrainTestRmBgAlign')

SELECTED_TEST = """47500_01_Normalyoungchild.png       
18495_01_Normalyoungchild.png         
74948_01_Normalyoungchild.png         
9121_01_Normalyoungchild.png""" 

SELECTED_TEST = [s.strip() for s in SELECTED_TEST.split()]


# ---------------------------------------------------------------------------- #

df = pd.read_csv(df)
img = os.listdir(img_path)

df = df[ df['name'].isin ( img ) ] # ! filter based on what was aligned correctly 

df = df[ ~df['name'].isin ( SELECTED_TEST ) ] # ! remove the selected test

normal_label = ['Normalyoungadult','Normalolderadult','Normalyoungchild','Normaladolescence','Normal2y']
df = df [ df['label'].isin(normal_label) ] # ! just in case. 

newdf = []
for i,l in enumerate(normal_label): 
  temp = df[df['label'] == l]
  temp = temp.sample(n=NUM_PICK,random_state=i).reset_index(drop=True)
  newdf.append(temp)



# ---------------------------------------------------------------------------- #
# ! combine all labels  
newdf = pd.concat(newdf)
newdf = newdf.reset_index(drop=True)
newdf = newdf.sample(frac=1,random_state=2022).reset_index(drop=True)

# ---------------------------------------------------------------------------- #
# ! replace path

newpath = [ re.sub('/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-BlankBackgroundCenter',img_path,p) for p in newdf['path'].values ]
newdf['path'] = newpath


# ---------------------------------------------------------------------------- #
# ! replace labels 
newdf['detail_label'] = newdf['label']
# newdf['label'] = 'Normal'
newdf['label'] = 'Unaffected'

# ---------------------------------------------------------------------------- #

if NUM_PICK == 500: 
  fold = [i%5 for i in range(newdf.shape[0])] 
  newdf ['fold'] = fold 
else: 
  newdf ['fold'] = 5 

# ---------------------------------------------------------------------------- #
# ! external data
df['is_ext'] = 1

# ---------------------------------------------------------------------------- #
#
newdf.to_csv(os.path.join(mainpath,headername+'-'+str(NUM_PICK)+'EachNormalAsUnaff.csv'),index=None)
