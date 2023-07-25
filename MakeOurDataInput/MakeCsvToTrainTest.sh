#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ---------------------------------------------------------------------------- # 

# ! ! keep background 

cd /data/duongdb/ClassifyManyFaceConditions/MakeOurDataInput
maindir=/data/duongdb/ManyFaceConditions12012022

use_previous_fold_csv='/data/duongdb/ManyFaceConditions05092022/Classify/ManyCondition+500Normal-AlignPix255-train.csv'

csvbasename=$maindir/Classify/ManyCondition+500Normal-AlignPix255 # ! output base name, we will append "train/test" to output

# ! test 
# don't need to pass in testcsv if we put everything into same folder 
# img_path=$maindir/TrimImg_align_255pix_testset
# # --normal_as_unaffected
# python3 MakeCsvFromImgPath.py --csv $csvbasename --img_path $img_path --istest --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' 

# ! test with Tobii images 
# don't need to pass in testcsv if we put everything into same folder 
img_path=$maindir/TrimImg_align_255pix_testset_add_tobii
# --normal_as_unaffected
python3 MakeCsvFromImgPath.py --csv $csvbasename --img_path $img_path --istest --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' --addname 'tobii'


# ! train
img_path=$maindir/TrimImg_align_255pix
# --normal_as_unaffected
python3 MakeCsvFromImgPath.py --csv $csvbasename --img_path $img_path --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' --normal_csv $maindir/Train-500EachNormal-align255pix.csv --use_previous_fold_csv $use_previous_fold_csv # -nobg

cd $maindir/Classify


# ---------------------------------------------------------------------------- # 

# ! ! remove background 

cd /data/duongdb/ClassifyManyFaceConditions/MakeOurDataInput
maindir=/data/duongdb/ManyFaceConditions12012022

csvbasename=$maindir/Classify/ManyCondition+500Normal-AlignPix255-no_bg # ! output base name, we will append "train/test" to output

# ! test 
# don't need to pass in testcsv if we put everything into same folder 
img_path=$maindir/TrimImg_no_bg_255pix_align_testset
# --normal_as_unaffected
python3 MakeCsvFromImgPath.py --csv $csvbasename --img_path $img_path --istest --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' 

# ! train
img_path=$maindir/TrimImg_no_bg_255pix_align
# --normal_as_unaffected
python3 MakeCsvFromImgPath.py --csv $csvbasename --img_path $img_path --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' --normal_csv $maindir/Train-500EachNormal-align255pix-nobg.csv 

cd $maindir/Classify



# ---------------------------------------------------------------------------- #

# ! python check test images are strictly unique from train images. 
import os,sys,re
import pandas as pd 
os.chdir('/data/duongdb/ManyFaceConditions12012022/Classify')
x1 = pd.read_csv('ManyCondition+500Normal-AlignPix255-train.csv') # this will be the train set 
x2 = pd.read_csv('ManyCondition+500Normal-AlignPix255-test.csv')

x1 = set ( x1['name'].values.tolist() ) 
x2 = set ( x2['name'].values.tolist() ) 

assert len ( x1.intersection(x2) ) == 0
assert len ( x2.intersection(x1) ) == 0

# ---------------------------------------------------------------------------- #

# ! make csv with only fake test images
# ! we may need to manual delete very poor quality images. 



# Normal        2500  
# 22q11DS        591  
# WS             531  
# Down           352  
# NS             325  
# BWS            308  
# KS             246  
# Unaffected     228  
# WHS            178  
# CdLS           120  
# RSTS1          105  
# PWS            104  
