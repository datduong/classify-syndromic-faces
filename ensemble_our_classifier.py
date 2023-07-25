import sys, os, re, pickle
import pandas as pd
import numpy as np
from glob import glob
import argparse

import OtherMetrics

def read_output(filename,args,colname='label'): 
    # we need to worry only about our iamges
    df = pd.read_csv(filename) # name,patient_id,sex,age_approx,anatom_site_general_challenge,label,benign_malignant,target,tfrecord,width,height,filepath,fold,is_ext,is_test,0,1,2,3,...
    df = df.sort_values(by=colname,ignore_index=True) # sort just to be consisent. 
    # p = r'({})'.format('|'.join(map(re.escape, args.labels))) # https://stackoverflow.com/questions/11350770/select-by-partial-string-from-a-pandas-dataframe
    # df = df[df[colname].str.contains(p)]
    df = df [ df[colname].isin(args.labels) ] 
    df = df.reset_index()
    print ('\nread in {} dim {}'.format(filename,df.shape[0]))
    # print (df)
    return df


def rm_lt50_average(prediction_array,num_labels): # @prediction_array is [[model1], [model2]...] for one single observation 
    counter = 0
    ave_array = np.zeros(num_labels)
    for array in prediction_array: 
        if max(array) > 0.2 : # skip if no prediction is over 0.5
            ave_array = ave_array + array 
            counter = counter + 1
    #
    return ave_array/counter 
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--labels', type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--keyword', default='test_on_fold_5_from_fold', type=str)
    parser.add_argument('--topk', default=2, type=int)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    args.labels = sorted ( args.labels.strip().split(',') ) 
    num_labels = len(args.labels)
    label_col_index = [str(i) for i in range(num_labels)] # ! col names are index, and not label names

    # ! 
    print ('model ', args.model_dir)
    
    # ! we need to rank a prediction probability for each condition
    # ! note that FOLD=5 IS DESIGNED AS THE TEST SET. 
    outputs = [read_output(csv,args) for csv in sorted(glob(os.path.join(args.model_dir, args.keyword+'*csv')))] # read each prediction

    final_df = outputs[0] # place holder 
    # print ('see first csv input', final_df)
    
    prediction_np = np.zeros((final_df.shape[0],num_labels))
    
    for index in range(final_df.shape[0]) : 
        # average ?
        prediction_array = []
        for df1 in outputs: # go over each csv output, and get the row
            array1 = list ( df1.loc[index,label_col_index] ) 
            prediction_array.append ( [float(a) for a in array1] ) # array of array of numbers. [ [1,2,3], [2,3,4] ... ]
        # 
        prediction_np[index] = rm_lt50_average(prediction_array,num_labels) # ensemble over many models for 1 observation 
    
    # print (prediction_np)
    # print (prediction_np.shape)
    final_df.loc[:,label_col_index] = prediction_np
    
    # need to recode the truth and prob labels. 
    diagnosis2idx = {value:index for index,value in enumerate(args.labels)}
    # our_label_index = [4, 5, 6, 7, 8, 9, 11]
    our_label_index = list ( np.arange(len(diagnosis2idx)) )
    
    final_df['true_label_index'] = final_df['label'].map(diagnosis2idx)
    
    PROBS = prediction_np.argmax(axis=1)
    final_df['predict_label_index'] = PROBS
    
    # print (PROBS)
    # print (PROBS.shape)
    for p in PROBS: 
        if p not in our_label_index: 
            print ('we predict something outside our list of labels')
            print (p)
    
    TARGETS = np.array ( list (final_df['true_label_index']) ) 
    # print (TARGETS)

    if args.output_name is None: 
        args.output_name = ''
        
    OtherMetrics.plot_confusion_matrix_manual_edit( prediction_np, TARGETS, diagnosis2idx, os.path.join(args.model_dir,args.output_name+'ensem_confusion_matrix' ), our_label_index )

    # ! rename col to disease names
    idx2diagnosis = {str(index):value for index,value in enumerate(args.labels)}
    final_df_fout = final_df.rename(columns=idx2diagnosis)
    final_df_fout = final_df_fout.drop(columns=['index','path','is_ext','fold','target'])
    final_df_fout.to_csv(os.path.join(args.model_dir,args.output_name+"final_prediction.csv"),index=None) # writeout

    # ! see recall (probably not important)
    # remove real normal images 
    final_df = final_df [ final_df['true_label_index'] != 6].reset_index(drop=True)
    prediction_np = final_df [ [str(i) for i in range(len(diagnosis2idx))] ].to_numpy() 
    y = np.array ( final_df['true_label_index'].values ).reshape(-1,1) # num_sample x 1 
    for k in np.arange(1,args.topk+1): 
        recall_ = OtherMetrics.recall_at_k_single_label(prediction_np, y, k)
        print ('without normal img, recall at ', k, ' is : ', recall_)

    # ! recall without Unaffected 
    final_df = final_df [ final_df['true_label_index'] != 9].reset_index(drop=True)
    prediction_np = final_df [ [str(i) for i in range(len(diagnosis2idx))] ].to_numpy() 
    y = np.array ( final_df['true_label_index'].values ).reshape(-1,1) # num_sample x 1 
    for k in np.arange(1,args.topk+1): 
        recall_ = OtherMetrics.recall_at_k_single_label(prediction_np, y, k)
        print ('without normal img and unaffected, recall at ', k, ' is : ', recall_)

