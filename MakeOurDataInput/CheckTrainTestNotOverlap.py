
import os,sys,re,pickle
import pandas as pd 
import numpy as np 

train_df = pd.read_csv('/data/duongdb/ManyFaceConditions04202022/Classify/ManyCondition+500Normal-AlignPix255-norm_as_unaff-train.csv')
test_df = pd.read_csv('/data/duongdb/ManyFaceConditions04202022/Classify/ManyCondition+500Normal-AlignPix255-norm_as_unaff-test.csv')

train_name = set ( train_df['name'].values ) 
test_name = set ( test_df['name'].values ) 

assert len ( train_name.intersection(test_name) ) == 0 



