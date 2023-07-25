
# ! create csv from fake img. 
# ! move everything into same folder. 

fout=/data/duongdb/ManyFaceConditions05092022/Stylegan3Model/Res256AlignPix255-0Normal-SkipAgeUp1-5/00000-stylegan2-Res256AlignPix255-0Normal-SkipAgeUp1-5-gpus2-batch64-gamma0.2048/network-snapshot-000887.pklInterpolate/T.8Static
mkdir $fout

for this in 0 1 2 3 4 5 6 7 8 9 10
do 
mv /data/duongdb/ManyFaceConditions05092022/Stylegan3Model/Res256AlignPix255-0Normal-SkipAgeUp1-5/00000-stylegan2-Res256AlignPix255-0Normal-SkipAgeUp1-5-gpus2-batch64-gamma0.2048/network-snapshot-000887.pklInterpolate/$this'T.8Static'/*png $fout
done

