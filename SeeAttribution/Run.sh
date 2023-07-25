


source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

for index in 0 1 2 3 4 5 6 7 8 9 10 11 
do

cd /data/duongdb/ClassifyManyFaceConditions/SeeAttribution

modeldir=/data/duongdb/ManyFaceConditions12012022/Classify/b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff/EvalTestImgLabelIndex$index

for condition_name in 22q11DS BWS CdLS Down KS NS PWS RSTS1 WHS Unaffected WS Normal
do 
python3 AverageAttrImg.py --model_dir $modeldir --fold 0,1,2,3,4 --folder_name '_test_Occlusion2.0' --condition_name $condition_name --keyword "_heatmappositive" # '_blended_heat_map_pos' _heatmappositive
done 

done 

cd $modeldir


# --keyword 'heatmappositive' --keyword '_attr_np' --use_attr_np --positive_np # ! may have to skip fold 

# ['22q11DS', 'BWS', 'CdLS', 'Down', 'KS', 'NS', 'PWS', 'RSTS1', 'Unaffected', 'WHS', 'WS']

# -------------------------------------------------------------------------------------------

# ! average all ATTRIBUTIONS INTO 1 SINGLE IMAGE FOR EACH DISEASE

cd /data/duongdb/ClassifyManyFaceConditions/SeeAttribution

image_path=/data/duongdb/WS22qOther_08102021/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter/EvalTrainTest/AverageAttr_Occlusion2

from_csv=/data/duongdb/WS22qOther_08102021/Classify/test+WS+22q11DS+Control+Normal+Split.csv

for keyword in 22q11DS BWS CdLS Down KS NS PWS RSTS1 WHS Unaffected WS Normal
do
  output_name='average_'$keyword
  python3 AverageAllAttrIn1Label.py --image_path $image_path --output_name $output_name --keyword $keyword --from_csv $from_csv 
  # --maptype heatmappositive
done 


# -------------------------------------------------------------------------------------------

# ! average all ATTRIBUTIONS INTO 1 SINGLE IMAGE FOR EACH DISEASE based on final prediction prob. 

for index in 0 1 2 3 4 5 6 7 8 9 10 11 
do

cd /data/duongdb/ClassifyManyFaceConditions/SeeAttribution

modeldir=/data/duongdb/ManyFaceConditions12012022/Classify/b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff/EvalTestImgLabelIndex$index/AverageAttr_test_Occlusion2.0

from_csv=/data/duongdb/ManyFaceConditions12012022/Classify/b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff/final_prediction.csv

# output_name=$index'_average_atleast0.2_'$maptype
# python3 AverageAllAttrIn1Label.py --image_path $modeldir --output_name $output_name --from_csv $from_csv --filter_col $index --filter_by_numeric 0.2 --maptype _blended_heat_map_pos

# output_name=$index'_average_true_'$maptype
# python3 AverageAllAttrIn1Label.py --image_path $modeldir --output_name $output_name --keyword $index --maptype _blended_heat_map_pos

output_name=$index'_average_true_atleast0.2_'$maptype
python3 AverageAllAttrIn1Label.py --image_path $modeldir --output_name $output_name --keyword $index --from_csv $from_csv --filter_col $index --filter_by_numeric 0.2 --maptype _blended_heat_map_pos

done 


# ---------------------------------------------------------------------------- #

mod=b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff
cd /cygdrive/c/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/$mod
scp -r $helix:/data/duongdb/ManyFaceConditions12012022/Classify/$mod/EvalTestImgLabelIndex* . 


#######

#######


mod=b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff
cd /cygdrive/c/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/$mod

for i in 11 # 0 1 2 3 4 5 6 7 8 9 10 11 
do 
# mkdir /cygdrive/c/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/$mod/EvalTestImgLabelIndex$i
cd /cygdrive/c/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/$mod/EvalTestImgLabelIndex$i
scp -r $helix:/data/duongdb/ManyFaceConditions12012022/Classify/$mod/EvalTestImgLabelIndex$i/Ave* EvalTestImgLabelIndex$i # EvalTestImgLabelIndex$i
done 

