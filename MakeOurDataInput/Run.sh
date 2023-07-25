# ---------------------------------------------------------------------------- #

# new conda? 
source /data/$USER/condaPy39/etc/profile.d/conda.sh
conda activate condaPy39
module load CUDA/11.3.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ---------------------------------------------------------------------------- #

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=6
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=12:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8 
# sbatch --partition=gpu --time=1-00:00:00 --gres=gpu:v100x:2 --mem=20g --cpus-per-task=20 


# ---------------------------------------------------------------------------- #

#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0


# ---------------------------------------------------------------------------- #
# ! need to realign WS and 22q? remove background of WS and 22q? 
WS22qPath=/data/duongdb/WS22qOther_08102021/TrimImg # !
resolution=512
datapath=/data/duongdb/ManyFaceConditions01312022/
codepath=$datadir/stylegan3-FaceSyndromes/FaceSyndromes/ManySyndromes # ! 
cd $codepath
python3 AlignImage.py --input_file_path $WS22qPath --output_file_path $datapath/Align$resolution'CenterWS22q' --output_size $resolution --centerface '0,0,512,512' --notblur > $codepath/align_log_background_many_conditions.txt
# ! remove background 
python3 RemoveBackground.py --imagepath $datapath/Align$resolution'CenterWS22q' --foutpath $datapath/Align$resolution'CenterRmBgWS22q' --colorcode 0 

# ! align first, then center? ---> avoid problem with false rm background, which is caught by alignment? 
WS22qPath=/data/duongdb/WS22qOther_08102021/TrimImg # !
resolution=512
datapath=/data/duongdb/ManyFaceConditions01312022/
codepath=$datadir/stylegan3-FaceSyndromes/FaceSyndromes/ManySyndromes # ! 
cd $codepath
python3 RemoveBackground.py --imagepath $WS22qPath --foutpath $datapath/'RmBgWS22q' --colorcode 0 
python3 AlignImage.py --input_file_path $datapath/'RmBgWS22q' --output_file_path $datapath/'RmBgAlignWS22q' --output_size $resolution --centerface '0,0,512,512' --notblur > $codepath/align_log_background_many_conditions.txt


# ------------------------------------------------------------------------------------------------------

cd /data/duongdb/ClassifyManyFaceConditions/MakeOurDataInput
maindir=/data/duongdb/ManyFaceConditions01312022

csv=$maindir/Classify/ManyCondition-NormalAsUnaff-Other-RmBgAlign-Easy.csv
img_path=$maindir/RmBgAlign9Cond+WS22qTest # ! test 

# python3 MakeCsvFold.py --csv $csv --img_path $img_path --istest --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' 

python3 MakeCsvFromImgPath.py --csv $csv --img_path $img_path --istest --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' --normal_csv $maindir/Test-25EachNormalAsUnaff.csv

csv=$maindir/Classify/ManyCondition-NormalAsUnaff-Other-RmBgAlign-Easy.csv 
img_path=$maindir/RmBgAlign9Cond+WS22qTrain # ! train

# python3 MakeCsvFold.py --csv $csv --img_path $img_path --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' 

python3 MakeCsvFromImgPath.py --csv $csv --img_path $img_path --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' --normal_csv $maindir/Train-500EachNormalAsUnaff.csv


# ------------------------------------------------------------------------------------------------------

# ! binary labels 
cd /data/duongdb/ClassifyManyFaceConditions/MakeOurDataInput
maindir=/data/duongdb/ManyFaceConditions01312022

csv=$maindir/ManyCondition-Normal-Other-RmBg-Easy.csv
img_path=$maindir/TestImgRmBgEasy # ! test 
python3 MakeCsvFold.py --csv $csv --img_path $img_path --istest --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' --binary_label

csv=$maindir/ManyCondition-Normal-Other-RmBg-Easy.csv 
img_path=$maindir/Align512CenterRmBg11CondEasy # ! train
python3 MakeCsvFold.py --csv $csv --img_path $img_path --disease 'WS,22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,WHS,Unaffected' --binary_label



# ---------------------------------------------------------------------------- #


z = """
 BWSSlide11.png                        
 BWSSlide63.png                        
 DownSlide12.png                       
 DownSlide60.png                       
 PWSSlide87.png                        
 RSTS1Slide3.png                       
 22q11DS_late213_22q11DSyoungchild.png 
 BWSSlide22.png                        
 CdLSSlide123.png                      
 DownSlide124.png                      
 KSSlide133.png                        
 NSSlide29.png                         
 PWSSlide44.png                        
 RSTS1Slide50.png                      
 UnaffectedSlide149.png                
 UnaffectedSlide150.png                
 UnaffectedSlide151.png                
 UnaffectedSlide152.png                
 UnaffectedSlide156.png                
 UnaffectedSlide165.png                
 WHSSlide153.png                       
 WS_inter25_WSyoungchild.png           
 18495_01_Normalyoungchild.png         
 47500_01_Normalyoungchild.png         
 74948_01_Normalyoungchild.png         
 9121_01_Normalyoungchild.png          
""".split()
z = [i.strip() for i in z]
p1 = '/data/duongdb/ManyFaceConditions01312022/Normal600SubsetTrainTestRmBgAlign'
p2 = '/data/duongdb/ManyFaceConditions01312022/RmBgAlign9Cond+WS22qTest'

import os
for i in z: 
  os.system ('mv ' + os.path.join(p1,i) + ' ' + os.path.join(p2,i)) 



for i in 18495_01_Normalyoungchild.png 47500_01_Normalyoungchild.png 74948_01_Normalyoungchild.png 9121_01_Normalyoungchild.png  
do 
echo 'train'
grep $i /data/duongdb/ManyFaceConditions01312022/Train-500EachNormal.csv
echo 'test'
grep $i /data/duongdb/ManyFaceConditions01312022/Test-20EachNormal.csv
done 

# ---------------------------------------------------------------------------- #
# ! check 
import os
set1 = set(os.listdir('/data/duongdb/ManyFaceConditions01312022/RmBgAlign9Cond+WS22qTrain'))  # Normal600SubsetTrainTestRmBgAlign
set2 = set(os.listdir('/data/duongdb/ManyFaceConditions01312022/RmBgAlign9Cond+WS22qTest'))

set1.intersection(set2)
