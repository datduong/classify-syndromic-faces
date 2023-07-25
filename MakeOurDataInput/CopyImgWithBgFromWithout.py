
import os,sys,re,pickle
import pandas as pd 
import numpy as np 

# ! some images fail to remove bg, so we based our "static" set wrt images_without_bg 

source_path_with_bg = '/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-AlignPix255' # has all 60k images 

final_path_with_bg = '/data/duongdb/ManyFaceConditions04202022/Normal500SubsetTrainTest-align255pix' # only images we want 
# os.mkdir(final_path_with_bg)

final_path_without_bg = '/data/duongdb/ManyFaceConditions04202022/Normal500SubsetTrainTest-align255pix-nobg' # images that successfully got bg removed


# for i in os.listdir(final_path_without_bg): 
#   os.system('scp ' + os.path.join(source_path_with_bg, i ) + ' ' + final_path_with_bg)


# ---------------------------------------------------------------------------- #

# create the csv, just need to replace name 

df = pd.read_csv('/data/duongdb/ManyFaceConditions04202022/Train-500EachNormal-align255pix-nobg.csv')
temp = [ re.sub(final_path_without_bg,final_path_with_bg,i) for i in df['path'].values]

df['path'] = temp
df.to_csv('/data/duongdb/ManyFaceConditions04202022/Train-500EachNormal-align255pix.csv', index=None)

