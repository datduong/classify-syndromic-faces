#!/bin/bash

# ---------------------------------------------------------------------------- #

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=12:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:2 --mem=20g --cpus-per-task=20 

# ---------------------------------------------------------------------------- #

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ---------------------------------------------------------------------------- #
# ! remove background of WS+22q
datadir=/data/duongdb/

WS22qPath=/data/duongdb/WS22qOther_08102021/TrimImg # ! used in previous paper

foutpath=/data/duongdb/ManyFaceConditions01312022/RmBgWS22q
foutpath2=/data/duongdb/ManyFaceConditions01312022/RmBgAlignWS22q


resolution=512
datapath=/data/duongdb/ManyFaceConditions01312022/
codepath=$datadir/stylegan3-FaceSyndromes/FaceSyndromes/ManySyndromes # ! 
cd $codepath
python3 RemoveBackground.py --imagepath $WS22qPath --foutpath $foutpath --colorcode 0 
python3 AlignImage.py --input_file_path $foutpath --output_file_path $foutpath2 --output_size $resolution --centerface '0,0,512,512' --notblur > $codepath/align_log_background_many_conditions.txt



# ---------------------------------------------------------------------------- #
# ! remove background of the other conditions 
datadir=/data/duongdb/

img_path=/data/duongdb/ManyFaceConditions01312022/TrimImg # ! used in previous paper

foutpath=/data/duongdb/ManyFaceConditions01312022/RmBg9Cond
foutpath2=/data/duongdb/ManyFaceConditions01312022/RmBgAlign9Cond

resolution=512
datapath=/data/duongdb/ManyFaceConditions01312022/
codepath=$datadir/stylegan3-FaceSyndromes/FaceSyndromes/ManySyndromes # ! 
cd $codepath
python3 RemoveBackground.py --imagepath $img_path --foutpath $foutpath --colorcode 0 
python3 AlignImage.py --input_file_path $foutpath --output_file_path $foutpath2 --output_size $resolution --centerface '0,0,512,512' --notblur > $codepath/align_log_background_many_conditions.txt


# ---------------------------------------------------------------------------- #
