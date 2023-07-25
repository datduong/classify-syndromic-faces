
import os,sys,re,pickle
import numpy as np 
import pandas as pd 


from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
from snorkel.labeling import LFAnalysis

# @L_train is surveys seen. @L_test doesn't exists, because people only see test set. 

# L_train = np.array([[1,-1,0],[-1,0,0],[-1,1,0]])
# Y_train = np.array([1,0,0])

cardinality=2

fin = '/data/duongdb/WS22qOther_08102021/QualtricOuput/22q11DS_per_img_by_each_doc_model_Standard.csv'  # WS_per_img_by_each_doc_model_Standard 22q11DS_per_img_by_each_doc_model_Standard

df = pd.read_csv(fin,header=None)
L_train = df.drop(columns=[0]).to_numpy() # ! load in predicted labels by 30 people 
Y_train = []
for y in list ( df[0] ) : 
    if 'Controls' in y: 
        Y_train.append(0)
    else: 
        Y_train.append(1)

#
Y_train = np.array ( Y_train ) # ! load in the true label 


# ! MAJORITY. simple max count on their real answer. 
majority_model = MajorityLabelVoter() # ! no need to train. 
majority_acc = majority_model.score(L=L_train, Y=Y_train, tie_break_policy="random")[
    "accuracy"
]

# ! model agreement/disagreement. 
label_model = LabelModel(cardinality=cardinality, verbose=True)
label_model.fit(L_train=L_train, Y_dev=Y_train, n_epochs=200, log_freq=100, seed=123, lr=0.05)

label_model_acc = label_model.score(L=L_train, Y=Y_train, tie_break_policy="random")[
    "accuracy"
]

probs_train = label_model.predict_proba(L=L_train)

# ! 

print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")

# label_model.get_weights()



# ---------------------------------------------------------------------------


LF_empirical_acc = LFAnalysis(L=L_train).lf_summary(Y_train)

keep = np.where ( LF_empirical_acc['Emp. Acc.'] > .6)[0]

L_train = L_train[ : , keep]



# ! MAJORITY. simple max count on their real answer. 
majority_model = MajorityLabelVoter() # ! no need to train. 
majority_acc = majority_model.score(L=L_train, Y=Y_train, tie_break_policy="random")[
    "accuracy"
]

# ! model agreement/disagreement. 
label_model = LabelModel(cardinality=cardinality, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123, lr=0.05)

label_model_acc = label_model.score(L=L_train, Y=Y_train, tie_break_policy="random")[
    "accuracy"
]

probs_train = label_model.predict_proba(L=L_train)

print (L_train.shape)

print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")
