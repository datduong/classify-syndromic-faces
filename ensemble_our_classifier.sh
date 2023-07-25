
# ----------------------------------------------------------------------------------------------------------------

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyManyFaceConditions

for modelname in b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff # b4ns448wlEqualss10lr1e-05dp0.2b64ntest1NormalNotAsUnaff b4ns448wl10ss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff b4ns448wl10ss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff-no_bg
do

cd /data/duongdb/ClassifyManyFaceConditions

modeldir="/data/duongdb/ManyFaceConditions12012022/Classify/"$modelname

labels='22q11DS,BWS,CdLS,Down,KS,NS,Normal,PWS,RSTS1,Unaffected,WHS,WS' # 'Affected,Unaffected' # '22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,Unaffected,WHS,WS'

python3 ensemble_our_classifier.py --model_dir $modeldir --labels $labels --topk 3

# python3 ensemble_our_classifier.py --model_dir $modeldir --labels $labels --topk 3 --keyword test-on-bg_from_fold --output_name test-on-bg

done 
cd $modeldir



# ---------------------------------------------------------------------------- # 
# /data/duongdb/ManyFaceConditions12012022/Classify/b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff
# without normal img, recall at  1  is :  0.6666666666666666
# without normal img, recall at  2  is :  0.8974358974358975
# without normal img, recall at  3  is :  0.9230769230769231
# without normal img and unaffected, recall at  1  is :  0.6206896551724138
# without normal img and unaffected, recall at  2  is :  0.8620689655172413
# without normal img and unaffected, recall at  3  is :  0.896551724137931


without normal img, recall at  1  is :  0.775
without normal img, recall at  2  is :  0.875
without normal img, recall at  3  is :  0.925
without normal img and unaffected, recall at  1  is :  0.7666666666666667
without normal img and unaffected, recall at  2  is :  0.8333333333333334
without normal img and unaffected, recall at  3  is :  0.9


# ----------------------------------------------------------------------------------------------------------------

# ! ensemble many different runs separately. 
z = """
b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter                       
b4ns448Wl10ss10lr3e-05dp0.2b64ntest1T0.6WS+22q11DS+Control+Normal+kimg10+target0.6+TransA+blankcenter 
b4ns448Wl10ss10lr3e-05dp0.2b64ntest1T0.6WS+22q11DS+Control+Normal+kimg10+target0.6+DiscA+blankcenter
b4ns448Wl10ss10lr3e-05dp0.2b64ntest1M1T0.6WS+22q11DS+Control+Normal+kimg10+target0.6+blankcenter   
b4ns448Wl10ss10lr3e-05dp0.2b64ntest1M0.75T0.6WS+22q11DS+Control+Normal+kimg10+target0.6+blankcenter
b4ns448Wl10ss10lr3e-05dp0.2b64ntest1M0.55T0.6AveWS+22q11DS+Control+Normal+kimg10+target0.6+blankcenter
b4ns448Wl10ss10lr3e-05dp0.2b64ntest1M0.75T0.6AveWS+22q11DS+Control+Normal+kimg10+target0.6+blankcenter
b4ns448Wl10ss10lr3e-05dp0.2b64ntest1M1T0.6AveWS+22q11DS+Control+Normal+kimg10+target0.6+blankcenter""".split()
' '.join(s.strip() for s in z)

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

cd /data/duongdb/ClassifyFaceConditions

for modelname in b4ns448wl10ss10lr3e-05dp0.2b64ntest1WS+22q11DS+Control+Normal+Whole+blankcenter
do
cd /data/duongdb/ClassifyFaceConditions
modeldir="/data/duongdb/ManyFaceConditions01312022/Classify/"$modelname
labels='Affected,Unaffected' # 'Affected,Unaffected' # '22q11DS,BWS,CdLS,Down,KS,NS,PWS,RSTS1,Unaffected,WHS,WS' 
python3 ensemble_our_classifier.py --model_dir $modeldir --labels $labels
done 
cd $modeldir


# ----------------------------------------------------------------------------------------------------------------


# ! copy images to local pc

mkdir /cygdrive/c/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/
for modelname in b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff-no_bg b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff # b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff 
do
mkdir /cygdrive/c/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/$modelname
cd /cygdrive/c/Users/duongdb/Documents/ManyFaceConditions12012022/Classify/$modelname
# scp -r $biowulf:/data/duongdb/ManyFaceConditions12012022/Classify/$modelname/*png .
scp -r $biowulf:/data/duongdb/ManyFaceConditions12012022/Classify/$modelname/*final*csv .
done 


