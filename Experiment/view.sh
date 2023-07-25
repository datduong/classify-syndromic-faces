
cd /data/duongdb/ManyFaceConditions04202022/Classify

for d in b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalAsUnaff b4ns448wl10ss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff b4ns448wlEqualss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff
do
  echo $d
  cat $d/log_9c_b4ns_448_eval.txt
done 



cd /data/duongdb/ManyFaceConditions04202022/Classify

for d in b4ns448wl10ss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff b4ns448wl10ss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff-no_bg
do
  echo $d
  grep 'Fold 2' $d/log_9c_b4ns_448_eval.txt
done 


