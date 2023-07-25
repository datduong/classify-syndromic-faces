import re,sys,os,pickle
from datetime import datetime
import time

# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=24
# sbatch --partition=gpu --time=4:00:00 --gres=gpu:p100:1 --mem=16g --cpus-per-task=24
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=10g --cpus-per-task=20
# sbatch --time=12:00:00 --mem=100g --cpus-per-task=24
# sinteractive --time=1:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12

script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! check model name
weight=WEIGHT
learningrate=LEARNRATE
imagesize=IMAGESIZE
schedulerscaler=ScheduleScaler 
dropout=DROPOUT

batchsize=64 

ntest=1 # ! we tested 1, and it looks fine at 1, don't need data aug during testing

kernel_type=9c_b4ns_$imagesize # ! this is experiment name

suffix=SUFFIX

# ! check if we use 60k or not
model_folder_name=b4ns$imagesize$imagetype'ss'$schedulerscaler'lr'$learningrate'dp'$dropout'b'$batchsize'ntest'$ntest$suffix 

maindir=/data/duongdb/ManyFaceConditions01312022/Classify

modeldir=$maindir/$model_folder_name 
mkdir $modeldir

logdir=$maindir/$model_folder_name 

# oofdir=$maindir/$model_folder_name/EvalDev
# oofdir=$maindir/$model_folder_name/EvalTrain # ! change if needed

oofdir=$maindir/$model_folder_name/EvalTestImg

cd /data/duongdb/ClassifyManyFaceConditions

# ! train

imagecsv=$maindir/pretrain-50knormal-gender.csv # ! train input 

python train.py --image_csv $imagecsv --kernel_type $kernel_type --image_size $imagesize --enet_type tf_efficientnet_b4_ns --use_amp --CUDA_VISIBLE_DEVICES 0,1 --model_dir $modeldir --log_dir $logdir --num_workers 8 --fold 'FOLD' --out_dim 2 --weighted_loss $weight --n_epochs 30 --batch_size $batchsize --init_lr $learningrate --scheduler_scaler $schedulerscaler --dropout $dropout --n_test $ntest 


"""

path = '/data/duongdb/ManyFaceConditions01312022'
os.chdir(path)


SUFFIX='pretrain-50knormal-gender'

counter=0

numberoflayers=0

for fold in [0]: # ,1,2,3,4
  for imagesize in [448]: # 448 512 768 640
    for weight in [10]: # 5,10,
      for schedulerscaler in [10]:
        for learn_rate in [0.00001]:  # 0.00001,0.00003  # we used this too, 0.0001
          for dropout in [0.2]:
            script2 = re.sub('WEIGHT',str(weight),script)
            script2 = re.sub('IMAGESIZE',str(imagesize),script2)
            script2 = re.sub('numberoflayers',str(numberoflayers),script2)
            script2 = re.sub('SUFFIX',str(SUFFIX),script2)
            script2 = re.sub('LEARNRATE',str(learn_rate),script2)
            script2 = re.sub('ScheduleScaler',str(schedulerscaler),script2)
            script2 = re.sub('FOLD',str(fold),script2)
            script2 = re.sub('DROPOUT',str(dropout),script2)
            now = datetime.now() # current date and time
            scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
            fout = open(scriptname,'w')
            fout.write(script2)
            fout.close()
            # 
            time.sleep( 1 )
            os.system('sbatch --partition=gpu --time=30:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=16 ' + scriptname )
            counter = counter + 1 

#
exit()


