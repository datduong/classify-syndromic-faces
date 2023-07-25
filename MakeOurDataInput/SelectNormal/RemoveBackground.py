import re,sys,os,pickle
from datetime import datetime
import time

script="""#!/bin/bash

# ---------------------------------------------------------------------------- #

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ---------------------------------------------------------------------------- #
# ! remove background of already aligned "extra normal" images 
datadir=/data/duongdb/

resolution=720
colorcode=255

# ! order of align or remove bg should not matter....

img_path=/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021
foutpath=/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-AlignPix$colorcode
foutpath2=/data/duongdb/FairFace/FairFace-aligned-60k-agegroup-06012021-AlignPix$colorcode-RmbgPix$colorcode

codepath=/data/duongdb/stylegan3-FaceSyndromes/FaceSyndromes/ManySyndromes # ! 
cd $codepath

# ! align
python3 AlignImage.py --input_file_path $img_path --output_file_path $foutpath --output_size $resolution --centerface '0,0,720,720' --colorcode $colorcode --notblur --enable_padding --start START --end END --skip_exist

# ! remove background 
# python3 RemoveBackground.py --imagepath $foutpath --foutpath $foutpath2 --colorcode $colorcode --start START --end END # ! white background 255 or black background pixel=0 ?



"""

# ---------------------------------------------------------------------------- #


# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=8:00:00 --gres=gpu:p100:1 --mem=6g --cpus-per-task=4 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:2 --mem=20g --cpus-per-task=20 

StartDict = {0:10000, 10000:20000, 20000:30000, 30000:40000, 40000:50000, 50000:70000}
counter = 0
for start,end in StartDict.items(): 
  script2 = re.sub('START',str(start),script)
  script2 = re.sub('END',str(end),script2)
  now = datetime.now() # current date and time
  scriptname = 'script'+str(counter)+'-'+now.strftime("%m-%d-%H-%M-%S")+'.sh'
  fout = open(scriptname,'w')
  fout.write(script2)
  fout.close()
  # 
  time.sleep( 1 )
  os.system('sbatch --partition=gpu --time=3:00:00 --gres=gpu:p100:1 --mem=6g --cpus-per-task=4 ' + scriptname )
  counter = counter + 1 


