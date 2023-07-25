import enum
import os,sys,re,pickle 
import pandas as pd
import numpy as np 

# ! move images ? why not. 

new_path = '/data/duongdb/ManyFaceConditions04202022/Normal500SubsetTrainTest-align255pix-nobg'

source_formatted_img_path = '/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-AlignPix255-RmbgPix255'

# ---------------------------------------------------------------------------- #

# ! we copy selected images here some months ago. now, based on this list, we copy the "new formats" to @new_path

# names_to_move = os.listdir('/data/duongdb/ManyFaceConditions01312022/Normal600SubsetTrainTestRmBgAlign') 
names_to_move = pd.read_csv('/data/duongdb/ManyFaceConditions01312022/Train-600EachNormal.csv')
names_to_move = names_to_move['name'].values


# ! with the new alignment code, some images may fail? 

df = pd.read_csv('/data/duongdb/ManyFaceConditions01312022/Classify/ManyCondition-Normal-Other-RmBgAlign-Easy-train.csv')
previous_used = [i for i in df['name'].values if "Normal" in i]

select_new = set(names_to_move)-set(previous_used)

SELECTED_TEST = """47500_01_Normalyoungchild.png       
18495_01_Normalyoungchild.png         
74948_01_Normalyoungchild.png         
9121_01_Normalyoungchild.png""".split()

SELECTED_TEST = [s.strip() for s in SELECTED_TEST]

select_new = set ( select_new ) - set (SELECTED_TEST)


# ! replace image. in trainset ???
# 54607_01_Normalyoungchild.png --> 81136_01_Normalyoungchild.png because new alignment code fail ?? 



# ---------------------------------------------------------------------------- #


if os.path.exists(new_path): 
  os.system('rm -rf ' + new_path) # ! safe 

# 
os.mkdir(new_path)


# ! better to copy over so we can zip/upload later.
not_exist = [] 
for i in previous_used: # ! only copy what was used in prelim experiments 
  if os.path.exists(os.path.join(source_formatted_img_path,i)): 
    os.system ( 'scp ' + os.path.join(source_formatted_img_path,i) + ' ' + os.path.join(new_path,i) )
    # pass
  else: 
    not_exist.append (i)

# 
print (len(not_exist)) # 54607_01_Normalyoungchild.png

# ---------------------------------------------------------------------------- #

add_these_instead = []
for index,value in enumerate(not_exist): 
  age_group = value.split('_')[-1] 
  pick1 = [i for i in select_new if (age_group in i) and os.path.exists(os.path.join(source_formatted_img_path,i)) ]      
  add_these_instead.append(pick1[0])
  select_new.remove(pick1[0])

#
for i in add_these_instead: 
  if os.path.exists(os.path.join(source_formatted_img_path,i)): 
    os.system ( 'scp ' + os.path.join(source_formatted_img_path,i) + ' ' + os.path.join(new_path,i) )
    # pass
  else: 
    not_exist.append (i)

# 

