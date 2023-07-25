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
# ! cut out white space ---- specialized powerpoint 
headfolder=/data/duongdb/ManyFaceConditions03032022
mkdir $headfolder/SurveyPics3_7_22_cleanTrimImg # ! all images will be in same folder, we need to run the @extract_code 
outfolder_name=/data/duongdb/ManyFaceConditions03032022/SurveyPics3_7_22_cleanTrimImg
codepath=$datadir/stylegan3-FaceSyndromes/FaceSyndromes/ManySyndromes # ! 
cd $codepath
for type in SurveyPics3_7_22_clean
do 
  python3 CropWhiteSpaceCenter.py --folder_name $headfolder/$type --padding 0 --outformat png --label SurveyPics3_7_22_clean --outfolder_name $outfolder_name > $type'trim_log.txt' # ! @png is probably best for ffhq style @'dummy' is a hack
done 
cd $headfolder/


# ---------------------------------------------------------------------------- #

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! remove background 
datadir=/data/duongdb/

img_path=/data/duongdb/ManyFaceConditions03032022/SurveyPics3_7_22_cleanTrimImg # ! used in previous paper

foutpath=/data/duongdb/ManyFaceConditions03032022/SurveyPics3_7_22_cleanTrimImgRmBg
foutpath2=/data/duongdb/ManyFaceConditions03032022/SurveyPics3_7_22_cleanTrimImgRmBgAlign

resolution=1024

datapath=/data/duongdb/ManyFaceConditions03032022/
codepath=$datadir/stylegan3-FaceSyndromes/FaceSyndromes/ManySyndromes # ! 
cd $codepath

python3 RemoveBackground.py --imagepath $img_path --foutpath $foutpath --colorcode 255 # ! white background 

python3 AlignImage.py --input_file_path $foutpath --output_file_path $foutpath2 --output_size $resolution --centerface '0,0,1024,1024' --notblur --whitebackground > $codepath/align_log_background_many_conditions.txt # --whitebackground



SurveyPics3_7_22_clean

# ---------------------------------------------------------------------------- #
