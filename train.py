import os, json
import time
import random
import argparse
import numpy as np
from numpy.lib.twodim_base import diag
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2

import apex
from apex import amp

from dataset import get_df, get_transforms, DatasetFromCsv
from models import Effnet_Face_Conditions, Resnest_Face_Conditions, Seresnext_Face_Conditions

from SoftLabelLoss import cross_entropy_with_probs
import OtherMetrics

# ---------------------------------------------------------------------------- #


def parse_comma_separated_list_as_dict(s):
    if isinstance(s, dict):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return dict()
    # 
    # jsonString = '{"a":54, "b": 28}'
    # aDict = json.loads(jsonString)
    return json.loads(s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel_type', type=str, required=True)
    parser.add_argument('--image_csv', type=str, default=None)
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--enet_type', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--init_lr', type=float, default=3e-5) # default=3e-5
    parser.add_argument('--out_dim', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=15)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model_dir', type=str, default='./weights')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--fold', type=str) # default='0'
    parser.add_argument('--n_meta_dim', type=str, default='512,128')
    
    # ! added
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_test', type=int, default=1, help='how many times do we flip images, 1=>no_flip, max=8')
    parser.add_argument('--scheduler_scaler', type=float, default=10)
    parser.add_argument('--dropout', type=float, default=0.2) # default=0.5
    parser.add_argument('--no_scheduler', action='store_true', default=False)
    parser.add_argument('--our_data', type=str, default=None)
    parser.add_argument('--celeb_data', type=str, default=None)
    parser.add_argument('--coco_data', type=str, default=None)
    parser.add_argument('--weighted_loss', type=parse_comma_separated_list_as_dict, default=None) # string input to have many weights on each kind of label
    parser.add_argument('--soft_label', action='store_true', default=False)
    parser.add_argument('--num_fc', type=int, default=0)
    parser.add_argument('--ret_vec_rep', action='store_true', default=False)

    parser.add_argument('--from_pretrain', action='store_true', default=False)
    parser.add_argument('--loaded_model', type=str, default=None)
    parser.add_argument('--save_bal_acc_after_ep', type=int, default=0)

    parser.add_argument('--sweep', action='store_true', default=False)
    
    args = parser.parse_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion, args=None):

    model.train()
    train_loss = []
    bar = tqdm(loader) # ! loader is a DataLoader 
    for (data, target, target_soft, _, _) in bar: # added path name and original resize image (only needed during attribution)

        # if args.soft_label: 
        #     target = target_soft
            
        optimizer.zero_grad()
        
        if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
        else:
            data, target = data.to(device), target.to(device)
            meta = None
            
        logits = model(data, x_meta=meta)
        if isinstance(logits, tuple) : 
            logits, _ = logits  
                  
        if args.soft_label: 
            loss = criterion(logits, target_soft.to(device),weight=args.weighted_loss)
        else:
            loss = criterion(logits, target)
                
        if not args.use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if args.image_size in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    return train_loss


def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0: # ! return original if I = 0
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, n_test=1, is_ext=None, get_output=False, criterion=None, args=None):

    model.eval()
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target, target_soft, _, _) in tqdm(loader): # added path name, and original resize
            
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
            else:
                data, target = data.to(device), target.to(device)
                meta = None
            
            # average over all the augmentation of test data
            logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
            probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
            for I in range(n_test):
                l = model(get_trans(data, I), x_meta=meta)
                if isinstance(l, tuple) : 
                    l, _ = l
                logits += l
                probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu()) # ! because we keep original @target, we can't override @target with @target_soft until later

            if args.soft_label: 
                loss = criterion(logits, target_soft.to(device),weight=args.weighted_loss)
            else:
                loss = criterion(logits, target)
            
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100. # ! global accuracy
        bal_acc = OtherMetrics.compute_balanced_accuracy_score(PROBS, TARGETS)
        bal_acc_ourdata = OtherMetrics.compute_balanced_accuracy_score(PROBS[is_ext == 0], TARGETS[is_ext == 0]) 
        return val_loss, acc, bal_acc, bal_acc_ourdata # ! let's not return vec rep during training.


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, criterion):

    if args.DEBUG:
        args.n_epochs = 5
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 5)
    else:
        df_train = df[df['fold'] != fold] # ! take out a fold and keep it as valid
        df_valid = df[df['fold'] == fold]
        print ('df input size {}'.format(df.shape[0]))
        print ('df_train input size after remove fold {} {}'.format(fold,df_train.shape[0]))
        print ('df_valid input size after remove fold {} {}'.format(fold,df_valid.shape[0]))

    # ! take out fold id=5, this is our testset id
    df_train = df_train[df_train['fold'] != 5]
    
    dataset_train = DatasetFromCsv(df_train, 'train', meta_features, transform=transforms_train, soft_label=args.soft_label)
    dataset_valid = DatasetFromCsv(df_valid, 'valid', meta_features, transform=transforms_val, soft_label=args.soft_label)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # ! check label count
    print ( 'final df train labels count\n {}'.format ( df_train['label'].value_counts() ) ) 
    print ( 'final df valid labels count\n {}'.format ( df_valid['label'].value_counts() ) )
    print ( 'train and dev data size {} , {}'.format(len(dataset_train), len(dataset_valid)) )

    model = ModelClass(
        args.enet_type,
        n_meta_features=n_meta_features,
        n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
        out_dim=args.out_dim,
        pretrained=True, 
        args=args
    )

    model = model.to(device)

    # ! loading in a model
    if args.loaded_model is not None: 
        print ('\nloading {}\n'.format(args.loaded_model))
        
        try:  # single GPU model_file    
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(args.loaded_model, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(args.loaded_model)
      
        except:  # multi GPU model_file
            if args.CUDA_VISIBLE_DEVICES is None:
                state_dict = torch.load(args.loaded_model, map_location=torch.device('cpu'))
            else: 
                state_dict = torch.load(args.loaded_model)
                
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

        # ! load model after get @state_dict
        strict_load = True 
        if args.from_pretrain: # ! the pretrained models has 4 labels. 
            strict_load = False
            print ('from pretrain, will remove last fc layer')
            temp_ = [] 
            for key in state_dict.keys():
                if 'myfc' in key: # ! 100k oct doesn't have meta-features
                    temp_.append(key)
            #
            print ('not load {}'.format(temp_))
            for key in temp_: 
                del state_dict[key] 

        # ! finally load, set strict=False if use @from_pretrain ?
        model.load_state_dict(state_dict, strict=strict_load)

        
    # ! send to multiple gpus... only works well if we have model.forward, don't change forward func.
    if DP:
        model = apex.parallel.convert_syncbn_model(model)

    acc_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_all_fold{fold}.pth')
    model_file_final = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')

    # ! our own label, we may add normal faces.
    bal_acc_ourdata_max = 0. 
    model_file_our = os.path.join(args.model_dir, f'{args.kernel_type}_ourlabel_fold{fold}.pth')
    
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)

    if not args.no_scheduler: 
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
        scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=args.scheduler_scaler, total_epoch=1, after_scheduler=scheduler_cosine)

    best_epoch = 0 # ! early stop 
    
    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
		# scheduler_warmup.step(epoch - 1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion=criterion, args=args)
        val_loss, acc, bal_acc, bal_acc_ourdata = val_epoch(model, valid_loader, n_test=args.n_test, is_ext=df_valid['is_ext'].values, criterion=criterion, get_output=False, args=args) 

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, bal_acc {(bal_acc):.6f}, bal_acc_ourdata {(bal_acc_ourdata):.6f}'
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        if not args.no_scheduler: 
            scheduler_warmup.step()    
            if epoch==2: scheduler_warmup.step() # bug workaround   
                
        if acc > acc_max: # ! save best model on all labels
            print('acc_max ({:.6f} --> {:.6f}). Saving model ...'.format(acc_max, acc))
            torch.save(model.state_dict(), model_file) ## @model is the same model in both cases, just eval them separately
            acc_max = acc
        
        # ! save model best for our data
        if (epoch > args.save_bal_acc_after_ep) and (bal_acc_ourdata > bal_acc_ourdata_max):
            print('bal_acc_ourdata_max ({:.6f} --> {:.6f}). Saving model ...'.format(bal_acc_ourdata_max, bal_acc_ourdata))
            torch.save(model.state_dict(), model_file_our)
            bal_acc_ourdata_max = bal_acc_ourdata
            best_epoch = epoch 
            
        # ! early stop based on acc. for our data
        if epoch - best_epoch > 20 : 
            break 

        if args.sweep: # ! https://docs.wandb.ai/guides/sweeps/quickstart
            metrics = { 'validation_loss': val_loss,
                        'acc': acc, 
                        'bal_acc': bal_acc,
                        'bal_acc_ourdata':bal_acc_ourdata}
            wandb.log(metrics)
        
    # ! end loop
    torch.save(model.state_dict(), model_file_final)


# ---------------------------------------------------------------------------- #


def main(df, criterion):

    transforms_train, transforms_val, _ = get_transforms(args.image_size) # don't need 3rd transform resize 

    folds = [int(i) for i in args.fold.split(',')]
    print ('\nfolds {}\n'.format(folds))
    for fold in folds: # ! run many folds
        run(fold, df, None, 0, transforms_train, transforms_val, criterion)
        

# ---------------------------------------------------------------------------- #


if __name__ == '__main__':

    args = parse_args()

    if not args.sweep: 
        with open(os.path.join(args.log_dir,'commandline_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if 'efficientnet' in args.enet_type:
        ModelClass = Effnet_Face_Conditions
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    set_seed(seed=args.seed) # ! set a seed, default to 0

    device = torch.device('cuda')

    # ! set up all args before calling "sweep"
    df, diagnosis2idx, _ = get_df(args.image_csv, soft_label=args.soft_label)

    if (args.weighted_loss is not None): # ! later add in @criterion so we can use weighted loss
        weight = torch.ones(args.out_dim) 
        for key,val in args.weighted_loss.items(): 
            weight[diagnosis2idx[key]] = val
        #
        args.weighted_loss = weight.to(device)
        print ('weight values {}'.format(args.weighted_loss))

    if not args.soft_label:     
        criterion = nn.CrossEntropyLoss(weight=args.weighted_loss)
    else: 
        criterion = cross_entropy_with_probs

    # ! set up sweep ? ------------------------------------------------------------------------
    if args.sweep:
        import wandb # ! https://docs.wandb.ai/guides/sweeps/quickstart
        wandb.init(config=args, allow_val_change=True) # @args pass into wandb.init 
        args = wandb.config # ! replace args # bug ? # ! https://github.com/wandb/client/issues/1737

    # ------------------------------------------------------------------------

    main(df, criterion) # ! train. 

    

