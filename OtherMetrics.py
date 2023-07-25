import os,sys,re,pickle 
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

# import torch
# from ignite.metrics import Accuracy, Precision, Recall, Fbeta


def plot_confusion_matrix_all_labels (prediction, y_true, diagnosis2idx, title, our_index, just_our_label=False): 

    labels = sorted ( list ( diagnosis2idx.keys() ) )
    output = confusion_matrix(y_true, prediction.argmax(axis=1)) ## ! take max as our best prediction
    sum_of_rows = output.sum(axis=1)
    normalized_array = output / sum_of_rows[:, np.newaxis] * 100 ## put back on 100 scale
    normalized_array = np.round (normalized_array,5)

    print (normalized_array.shape)
    print (normalized_array)

    if not just_our_label: 
        df_cm = pd.DataFrame(normalized_array, 
                            index = [i for i in labels],
                            columns = [i for i in labels]).astype(float).round(3)
        plt.figure(figsize=(12,12))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".1f", cbar=False) # font size
        plt.savefig(title+'.png')

    # ! do only our conditions 
    labels = [ lab if lab!='EverythingElse' else 'Other' for lab in labels]
    labels = [ labels[i] for i in our_index ]
    normalized_array = normalized_array[our_index,:][:,our_index] ## get our conditions
    df_cm = pd.DataFrame(normalized_array, 
                         index = [i for i in labels],
                         columns = [i for i in labels]).astype(float).round(3)
    plt.figure(figsize=(12,12))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".1f", cbar=False) # font size
    plt.savefig(title+'_ourlabels.png')
    pickle.dump(normalized_array,open(title+'_ourlabels.np','wb')) # later we will average over many folds


def plot_confusion_matrix_manual_edit (prediction, y_true, diagnosis2idx, title, our_index, figsize=16): 
    # ! different from @plot_confusion_matrix_all_labels, because we need to manual edit:
    # @figsize for publication 
    # @annot_kws ... and so forth ... 

    output = confusion_matrix(y_true, prediction.argmax(axis=1)) ## ! take max as our best prediction
    sum_of_rows = output.sum(axis=1)
    normalized_array = output / sum_of_rows[:, np.newaxis] * 100 ## put back on 100 scale
    normalized_array = np.round (normalized_array,5)
    print (normalized_array)

    # ! do only our conditions 
    temp = list ( set(prediction.argmax(axis=1)) )
    our_index = list ( set ( our_index + temp ) ) # combine the label index
    labels = sorted ( [ k for k,val in diagnosis2idx.items() if val in our_index ] ) # ! we misclassify something in our dataset as ISIC condition
    labels = [ lab if lab!='EverythingElse' else 'Other' for lab in labels]
    print (labels)
    
    df_cm = pd.DataFrame(normalized_array, 
                         index = [i for i in labels],
                         columns = [i for i in labels]).astype(float).round(3)
    if len(labels) <= 6: 
        figsize = 6
    elif len(labels) >= 15: 
        figsize = 20
    plt.figure(figsize=(figsize,figsize))
    sn.set_context("talk", font_scale=1)
    # sn.set(font_scale=1.4) # for label size
    ax = plt.axes()
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt=".1f", cbar=False, ax=ax) # font size
    ax.set_title('Confusion matrix')
    plt.xticks(rotation=45)
    plt.savefig(title+'_ourlabels.png')
    pickle.dump(normalized_array,open(title+'_ourlabels.np','wb')) # later we will average over many folds


def compute_balanced_accuracy_score (prediction,target): # may not need this, but we want sklearn to use pytorch format, @prediction comes before @target 
    return balanced_accuracy_score (target, prediction.argmax(axis=1))


def recall_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / num true labels
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]
    # get recall at k for each example
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        denom = y[i,:].sum()
        vals.append(num_true_in_top_k / float(denom))
    #
    vals = np.array(vals)
    vals[np.isnan(vals)] = 0.
    return np.mean(vals)


def precision_at_k(yhat_raw, y, k):
    # num true labels in top k predictions / k
    sortd = np.argsort(yhat_raw)[:,::-1]
    topk = sortd[:,:k]
    # get precision at k for each example
    vals = []
    for i, tk in enumerate(topk):
        if len(tk) > 0:
            num_true_in_top_k = y[i,tk].sum()
            denom = len(tk)
            vals.append(num_true_in_top_k / float(denom))
    return np.mean(vals)


def topk_accuracy(yhat_raw, target, topk=(1,)):
    metrics = {} # dict
    for k_i in topk:
        rec_at_k = recall_at_k(yhat_raw, target, k_i)
        metrics['rec_at_%d' % k_i] = rec_at_k
        prec_at_k = precision_at_k(yhat_raw, target, k_i)
        metrics['prec_at_%d' % k_i] = prec_at_k
        metrics['f1_at_%d' % k_i] = 2*(prec_at_k*rec_at_k)/(prec_at_k+rec_at_k)
    return metrics


def recall_at_k_single_label (yhat_raw, y, k): 
    # @yhat_raw is prob matrix, num_sample x label
    # @y is num_sample x 1 , eg. [[1],[2],[3]] not-1-hot, it is label index. 
    top_index = np.argsort(yhat_raw * -1 , axis=1) # largest, but default is to short smallest first 
    top_index_k = top_index [:,0:k]
    correct = 0.0 
    for i in range(top_index_k.shape[0]): 
        if y[i][0] in top_index_k[i,]: 
            correct = correct + 1 
    # 
    correct = correct / top_index_k.shape[0]
    return correct 
