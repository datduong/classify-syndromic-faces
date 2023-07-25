
cd /data/duongdb/ClassifyManyFaceConditions/WandbSweep
for num in 1 2 3 4
do
wandb sweep sweep$num.yaml
done 

# 


# ---------------------------------------------------------------------------


# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=12g --cpus-per-task=24
# sbatch --partition=gpu --time=4:00:00 --gres=gpu:p100:1 --mem=16g --cpus-per-task=24
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:p100:2 --mem=10g --cpus-per-task=20
# sbatch --time=12:00:00 --mem=100g --cpus-per-task=24
# sinteractive --time=3:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8

script = """#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0 # ! newest version at the time
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0


COMMAND 

"""

import re,sys,os,pickle
from datetime import datetime
import time

COMMAND = ['wandb agent datdddd/ClassifyManyFaceConditions/quu1hwpt', 
           'wandb agent datdddd/ClassifyManyFaceConditions/cl105t00',
           'wandb agent datdddd/ClassifyManyFaceConditions/3ehyghc0',
           'wandb agent datdddd/ClassifyManyFaceConditions/zssfebue',
          #  'wandb agent datdddd/ClassifyManyFaceConditions/b80zmauw'
           ]

path = '/data/duongdb/ManyFaceConditions01312022'
os.chdir(path)

for counter,command in enumerate(COMMAND): 
  now = datetime.now() # current date and time
  scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
  fout = open(scriptname,'w')
  script2 = re.sub('COMMAND',str(command),script)
  fout.write(script2)
  fout.close()
  # 
  time.sleep( 1 )
  os.system('sbatch --partition=gpu --time=40:00:00 --gres=gpu:v100x:1 --mem=8g --cpus-per-task=8 ' + scriptname )
  counter = counter + 1 
                
#

exit() 

