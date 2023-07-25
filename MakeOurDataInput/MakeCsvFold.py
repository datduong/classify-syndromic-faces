
from operator import index
import os,sys,re,pickle
import pandas as pd 
import numpy as np 
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, default=None)
parser.add_argument("--img_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--istest", action='store_true', default=False)
parser.add_argument("--disease", type=str, default=None)
parser.add_argument("--binary_label", action='store_true', default=False)


args = parser.parse_args()

args.disease = sorted ( args.disease.split(',') ) 

# ----------------------------------------------------------------------------------------------------------------

# check not overlap 
# import os
# x1 = set ( os.listdir('/data/duongdb/ManyFaceConditions01312022/Align512CenterRmBg11CondEasy') ) 
# x2 = set ( os.listdir('/data/duongdb/ManyFaceConditions01312022/TestImgRmBgEasy') )
# len(x1.intersection(x2)) == 0

# ----------------------------------------------------------------------------------------------------------------

np.random.seed(seed=args.seed)

# ! random scramble
# ! add in fold index

if args.istest: 
  fold = [5]
else:
  fold = [0,1,2,3,4] 

# ! 
args.img_path = args.img_path.split(',')
print ('img path', args.img_path)
image_name = []
for f in args.img_path: 
  print (f)
  image_name = image_name + os.listdir(f) # ! create csv from image folder, but csv has to exists already. 

# 
print ('image count', len(image_name))

# overlap with some existing df
df = pd.read_csv(args.csv)
df = df[df['name'].isin(image_name)]
df = df.reset_index(drop=True)

# ! random sample 
df = df.sample(frac=1,random_state=args.seed).reset_index(drop=True)

# ! test is fold 5, train is random split into 0-4
if args.istest: 
  addname = '-test.csv'
  fold = [5] * df.shape[0]
  # ! replace path 
  temp_ = [ os.path.join( args.img_path[0], f.split('/')[-1] ) for f in list(df['path']) ] # pass in @img_path as list
  df['path'] = temp_ 
else: 
  addname = '-train.csv'
  fold = []
  for i in range(df.shape[0]): 
    fold.append ( i%5 ) # mod 5 folds

# ! need # name,path,label,fold,is_ext
df['is_ext'] = 0 

# ! label 
label = []
for l in df['label'].values: 
  for disease in args.disease: 
    if bool(re.findall('^'+disease, l) ): 
      label.append(disease)

df['detail_label'] = df['label'].values

if args.binary_label: 
  temp_ = ['Unaffected' if l=='Unaffected' else 'Affected' for l in label ]
  df['label'] = temp_
else: 
  df['label'] = label

# assigned fold
df['fold'] = fold 

print ('df count', df.shape)
if args.binary_label: 
  addname = '-binary'+addname

df.to_csv(re.sub(r'\.csv','',args.csv)+addname,index=None)

