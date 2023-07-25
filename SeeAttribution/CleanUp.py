

import os,sys,re
import numpy as np 


mainpath = '/data/duongdb/ManyFaceConditions05092022/Classify/b4ns448wl10ss10lr3e-05dp0.2b64ntest1NormalNotAsUnaff'

basename = 'EvalTestImgLabelIndex'

for i in np.arange(12): 
  thispath = os.path.join(mainpath, basename+str(int(i)))
  for f in [0,1,2,3,4]: 
    print ('delete ', os.path.join(thispath, str(f)+'_test_Occlusion*'))
    os.system('rm -rf ' + os.path.join(thispath, str(f)+'_test_Occlusion*') ) 


#
