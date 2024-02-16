import re,sys,os,pickle
from datetime import datetime
import time
import numpy as np 


script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ---------------------------------------------------------------------------- # 

# ! check model name
weight=WEIGHT
learningrate=LEARNRATE
imagesize=IMAGESIZE
schedulerscaler=ScheduleScaler 
dropout=DROPOUT

batchsize=64 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=b4ns_$imagesize # ! this is experiment name

suffix=SUFFIX

# ! base-name of model output 
model_folder_name=b4ns$imagesize$imagetype'wl'$weight'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest$suffix 

maindir=/data/duongdb/ManyFaceConditions12012022/Classify

modeldir=$maindir/$model_folder_name 
mkdir $modeldir

logdir=$maindir/$model_folder_name 

oofdir=$maindir/$model_folder_name/EvalTestImg

cd /data/duongdb/ClassifyManyFaceConditions

# ---------------------------------------------------------------------------- #

# ! train

loaded_model=/data/duongdb/ManyFaceConditions12012022/Classify/b4ns448ss10lr1e-05dp0.2b64ntest1pretrain-50knormal-gender/9c_b4ns_448_best_all_fold0.pth

imagecsv=$maindir/ManyCondition+500Normal-AlignPix255-train.csv # ! train input 

python train.py --image_csv $imagecsv --kernel_type $kernel_type --image_size $imagesize --enet_type tf_efficientnet_b4_ns --use_amp --CUDA_VISIBLE_DEVICES 0 --model_dir $modeldir --log_dir $logdir --num_workers 8 --fold 'FOLD' --out_dim OUTDIM --weighted_loss 'DICT_LOSS_SCALE' --n_epochs 30 --batch_size $batchsize --init_lr $learningrate --scheduler_scaler $schedulerscaler --dropout $dropout --n_test $ntest --from_pretrain --loaded_model $loaded_model


# ---------------------------------------------------------------------------- #

# ! eval on our test set

imagecsv=$maindir/ManyCondition+500Normal-AlignPix255tobii-test.csv # ! test input 

python evaluate.py --image_csv $imagecsv --kernel_type $kernel_type --model_dir $modeldir --log_dir $logdir --image_size $imagesize --enet_type tf_efficientnet_b4_ns --oof-dir $oofdir --batch_size 64 --num_workers 4 --fold 'FOLD' --out_dim OUTDIM --CUDA_VISIBLE_DEVICES 0 --dropout $dropout --do_test --n_test $ntest

# ---------------------------------------------------------------------------- #

# ! look at pixel

# oofdir=$maindir/$model_folder_name/EvalTestImgLabelIndexATTR_INDEX # ! do this for each LABEL, it will take a lot of time to finish. 

# imagecsv=$maindir/ManyCondition+500Normal-AlignPix255-test.csv # ! test input 

# imagecsv=$maindir/ManyCondition+500Normal-AlignPix255tobii-test.csv

# for condition in ATTRIBUTELABEL
# do

# python evaluate.py --image_csv $imagecsv --kernel_type $kernel_type --model_dir $modeldir --log_dir $logdir --image_size $imagesize --enet_type tf_efficientnet_b4_ns --oof-dir $oofdir --batch_size 1 --num_workers 8 --fold 'FOLD' --out_dim OUTDIM --dropout $dropout --n_test $ntest --attribution_keyword $condition --outlier_perc 2 --attribution_model Occlusion --do_test --attr_label_index ATTR_INDEX 

# done

"""

# ---------------------------------------------------------------------------- #

path = '/data/duongdb/ManyFaceConditions12012022'
os.chdir(path)

# ---------------------------------------------------------------------------- #

# ! weigh the images, so each condition has almost uniform weights 
DICT_LOSS_SCALE = '{"22q11DS":4, "BWS":8, "CdLS":21, "Down":7, "KS":10, "NS":8, "PWS":24, "RSTS1":24, "Unaffected":11, "WHS":14, "WS":5}' 

# 22q11DS	591	4
# BWS	308	8
# CdLS	120	21
# Down	352	7
# KS	246	10
# Normal	2500	1
# NS	327	8
# PWS	104	24
# RSTS1	105	24
# Unaffected	228	11
# WHS	178	14
# WS	529	5

# 22q11DS	591	4.230118443	4
# BWS	308	8.116883117	8
# CdLS	120	20.83333333	20
# Down	352	7.102272727	7
# KS	246	10.16260163	10
# Normal	2500	1	1
# NS	325	7.692307692	7
# PWS	104	24.03846154	24
# RSTS1	105	23.80952381	23
# Unaffected	228	10.96491228	10
# WHS	178	14.04494382	14
# WS	531	4.708097928	4


# ---------------------------------------------------------------------------- #

# ! change name 
SUFFIX = 'NormalNotAsUnaff' # 'NormalNotAsUnaff' 'NormalAsUnaff' # normal is treated same as unaffected. 
OUTDIM = 12
if ("NormalAsUnaff" in SUFFIX): 
  OUTDIM = 11
  DICT_LOSS_SCALE ['Unaffected'] = 1

# ---------------------------------------------------------------------------- #

counter = 0
label_index = np.arange ( OUTDIM ) # ! MAY WANT TO EXCLUDE SOME LABELS 

# for ATTR_INDEX in label_index : # ! if run attribution 
#   for ATTRIBUTELABEL in 'WS,22q11DS,NS,BWS,CdLS,Down,KS,PWS,RSTS1,WHS,Unaffected,Normal'.split(',') : # ! if run attribution 
    # 'BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected,WS,22q11DS,Normal'.split(','): 
    # 'Unaffected'.split(','): #,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected,WS,22q11DS
for fold in [0,1,2,3,4]: # 0,1,2,3,4
  for imagesize in [448]: # 448 512 768 640 # ! see which image size may work best, 448 seems to be best
    for weight in ['Equal']: # ! we can use different types of naming system for the sample weights
      for schedulerscaler in [10]:
        for learn_rate in [0.00003]:  # 0.00001,0.00003  # ! tune learning rate, 0.00003 works best 
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            script2 = re.sub('DICT_LOSS_SCALE',str(DICT_LOSS_SCALE),script2)
            script2 = re.sub('OUTDIM',str(OUTDIM),script2)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('SUFFIX',str(SUFFIX),script2)
            script2 = re.sub('LEARNRATE',str(learn_rate),script2)
            script2 = re.sub('ScheduleScaler',str(schedulerscaler),script2)
            script2 = re.sub('FOLD',str(fold),script2)
            script2 = re.sub('DROPOUT',str(dropout),script2)
            # !
            # script2 = re.sub('ATTRIBUTELABEL',str(ATTRIBUTELABEL),script2) # ! if run attribution 
            # script2 = re.sub('ATTR_INDEX',str(ATTR_INDEX),script2)
            # 
            now = datetime.now() # current date and time
            scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
            fout = open(scriptname,'w')
            fout.write(script2)
            fout.close()
            # 
            time.sleep( 0.5 )
            # os.system('sbatch --partition=gpu --time=10:00:00 --gres=gpu:v100x:1 --mem=8g --cpus-per-task=8 ' + scriptname )
            os.system('sbatch --partition=gpu --time=00:40:00 --gres=gpu:p100:1 --mem=6g --cpus-per-task=8 ' + scriptname )
            # ! 
            # if (ATTRIBUTELABEL == 'Unaffected') or (ATTRIBUTELABEL == 'Normal'): # ! if run attribution 
            #   os.system('sbatch --time=6:30:00 --mem=64g --cpus-per-task=12 ' + scriptname )
            # else: 
            #   os.system('sbatch --time=4:30:00 --mem=64g --cpus-per-task=12 ' + scriptname )
            counter = counter + 1 

    #
exit()





