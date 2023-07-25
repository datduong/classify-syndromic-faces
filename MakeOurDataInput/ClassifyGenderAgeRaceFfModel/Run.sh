#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/FairFace

# ! there can be better models to predict gender/age/race... we can try those later. 

# ! make input 
maindir=/data/duongdb/ManyFaceConditions08172022/
img_dir=$maindir/TrimImg_no_bg_0pix_align/
output=$maindir/TrimImg_no_bg_0pix_align08172022Input.csv

python3 WsData/MakeInput.py --img_dir $img_dir --output $output 

# ! predict
outputfinal=$maindir/TrimImg_no_bg_0pix_align08172022_auto_age_gender_race.csv

python3 predict_to_csv.py --csv $output --output $outputfinal
rm -rf detected_faces # ! clean up



# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=16:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:2 --mem=20g --cpus-per-task=20 
