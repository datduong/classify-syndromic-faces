

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
parser.add_argument("--normal_as_unaffected", action='store_true', default=False)
parser.add_argument("--disease", type=str, default=None)
parser.add_argument("--binary_label", action='store_true', default=False)
parser.add_argument("--addname", type=str, default=None)
parser.add_argument("--normal_csv", type=str, default=None)
parser.add_argument("--use_previous_fold_csv", type=str, default=None) # ! use fold number from an older experiment to be consistent


args = parser.parse_args()

args.disease = sorted ( args.disease.split(',') ) 
if args.addname is None: 
  args.addname = '' 

if args.normal_as_unaffected: 
  args.addname = args.addname + '-norm_as_unaff'

# ----------------------------------------------------------------------------------------------------------------

# check not overlap 
# import os
# x1 = set ( os.listdir('/data/duongdb/ManyFaceConditions01312022/Align512CenterRmBg11CondEasy') ) 
# x2 = set ( os.listdir('/data/duongdb/ManyFaceConditions01312022/TestImgRmBgEasy') )
# len(x1.intersection(x2)) == 0

# ----------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------- #
# ! use previous fold assignment ?
if args.use_previous_fold_csv is not None: 
  previous_fold = pd.read_csv(args.use_previous_fold_csv)
  previous_fold_assignment = dict(zip(previous_fold['name'].tolist(), previous_fold['fold'].tolist()))
  print ('check previous fold')
  print (previous_fold_assignment['KSSlide16.png'])

# ---------------------------------------------------------------------------- #

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
image_count = len(image_name)
print ('image count', image_count)

# ! make dict 
df = {'name':[],
      'path':[],
      'label':[],
      'is_ext':[]}

df['name'] = image_name

# ! replace path 
df['path'] = [ os.path.join( args.img_path[0], f.split('/')[-1] ) for f in image_name ] 

# ! need # name,path,label,fold,is_ext
df['is_ext'] = [0] * image_count 

# ! label
label = []
for l in image_name: 
  #
  if 'Normal' in l: 
    if args.normal_as_unaffected: 
      label.append('Unaffected') # ! anything else Unaffected? 
    else: 
      label.append('Normal')
    #
    continue
  #
  for disease in args.disease: 
    if bool(re.findall('^'+disease, l) ): 
      label.append(disease)
  

if args.binary_label: 
  df['label'] = ['Unaffected' if l=='Unaffected' else 'Affected' for l in label ]
else: 
  df['label'] = label

# ! random sample 
df = pd.DataFrame.from_dict(df)
if not args.istest: 
  df = df.sample(frac=1,random_state=args.seed).reset_index(drop=True)

# ! test is fold 5, train is random split into 0-4
if args.istest: 
  addname = args.addname+'-test.csv'
  fold = [5] * image_count
else: 
  addname = args.addname+'-train.csv'
  fold = []
  all_image_name = df['name'].tolist()
  for i in range(image_count): 
    if args.use_previous_fold_csv is None: # ! random assign
      fold.append ( i%5 ) # mod 5 folds
    else: 
      if all_image_name[i] in previous_fold_assignment: # ! take the previous fold assignment 
        fold.append (previous_fold_assignment[all_image_name[i]])
      else: 
        fold.append (i%5) # new image just get a random assignment 

# assigned fold
df['fold'] = fold 


# ---------------------------------------------------------------------------- #
# ! add normal? 
if args.normal_csv is not None: 
  normal_df = pd.read_csv(args.normal_csv)
  if args.normal_as_unaffected: 
    normal_df['label'] = 'Unaffected'
  #
  df = pd.concat( [df, normal_df] )
  # 
  df = df.reset_index(drop=True)
  


# ---------------------------------------------------------------------------- #

if args.binary_label: 
  addname = '-binary'+addname


print ('df count', df.shape)
print ('df count', df.shape)

# 

print ( df ['label'].value_counts() ) 

df.to_csv(re.sub(r'\.csv','',args.csv)+addname,index=None)


