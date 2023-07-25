import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
import re, pickle
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def GetAttributionPlot (transformed_img, output, img_path, original_resize, attribution_model, fold, attribution_name='_', true_label_index=None, args=None, nn_model=None, skip_done=False): 

  # @attribution_name "integrated gradient"
  # @transformed_img batch x 3 x h x w (because RGB), after data augmentation, not just resize

  print ('these images ')
  for p in img_path: print (p)
  
  if args.attr_top_label: 
    prediction_score, pred_label_idx = torch.topk(output, 1) # take top prediction, # ! notice @prediction_score is not used
    print ('topk label index', pred_label_idx) # ! batch x 1 array ! looks like this tensor([1])
  elif args.attr_label_index > -1: 
    pred_label_idx = torch.LongTensor([args.attr_label_index]).repeat(output.shape[0]) # ! batch x 1 array... MUST BE INT
    print ('specific label index', pred_label_idx)
  else: 
    pred_label_idx = true_label_index # ! looks like this: true label index tensor([1])
    print ('true label index', pred_label_idx)

  temp_ = "_test_" if args.do_test else "_" 
  temp_ = temp_ + args.attribution_model
  if args.noise_tunnel: 
    temp_ = temp_ + '_noisetunnel'
  
  folder_path = os.path.join ( args.oof_dir, str(fold) + temp_ + str(args.outlier_perc)) # save images in its own fold
  if not os.path.exists (folder_path):
    os.mkdir(folder_path)

  # ! @n_steps will determine how much memory we need, low--> bad approx to the intergral, high--> too much mem
  # ! baseline is black image. We may try different baselines.
  if args.attribution_model == 'Occlusion': 
    attributions_ig = attribution_model.attribute(transformed_img, target=pred_label_idx, strides = (3, 10, 10), sliding_window_shapes=(3,20, 20) )
  else: 
    if not args.noise_tunnel: 
      attributions_ig = attribution_model.attribute(transformed_img, target=pred_label_idx, n_steps=75)
    else: 
      attributions_ig = attribution_model.attribute(transformed_img, target=pred_label_idx, nt_samples=5, nt_type='smoothgrad_sq')
  
  # default_cmap = LinearSegmentedColormap.from_list('custom blue', # white-->black color gradient scale
  #                                                 [ (0, '#ffffff'),
  #                                                   (0.5, '#000000'),
  #                                                   (1, '#000000')], N=256)

  default_cmap_black_white = LinearSegmentedColormap.from_list('custom black', # white-->black color gradient scale
                                                  [ (0, '#ffffff'),
                                                    (1, '#000000')], N=256)

  # we can only plot each image one at a time

  for b in range(transformed_img.shape[0]): 

    attribution_np = attributions_ig[b].squeeze().cpu().detach().numpy() # ! c x h x w ? 
    attribution_np_dim_hwc = np.transpose(attribution_np, (1,2,0)) # ! (224, 224, 3), h x w x chanels

    this_image_name = img_path[b].split('/')[-1]

    # ! check if file exists, and then skip. note we will always use batch=0
    if skip_done: 
      outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_heatmappositive.jpg', this_image_name)
      if os.path.exists(os.path.join(folder_path, outpath)):
        return 0
    
    # ! get embedding of @attribution_np? attribution_np is not between 0-1 as in standard images (do we care?)
    if args.attr_np_as_vec: 
      x = nn_model(attributions_ig[b].unsqueeze(0)) # batch x 3 x size x size. # @nn_model can be EfficientNet, Resnet ...
      x = x.squeeze().cpu().detach().numpy()
      outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_attr_as_vec.pickle', this_image_name)
      pickle.dump ( x, open ( os.path.join(folder_path, outpath), 'wb' ) ) 
      # save the np array 
      outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_attr_np.pickle', this_image_name)
      pickle.dump ( attribution_np, open ( os.path.join(folder_path, outpath), 'wb' ) ) 


    # ! heatmap overlap on gray scale @blended_heat_map
    image = viz.visualize_image_attr( attribution_np_dim_hwc, # ! (224, 224, 3), h x w x chanels
                                      original_resize[b].detach().numpy(), # @original_resize is tensor, even when we kept it as numpy, so we have to convert to np
                                      method='blended_heat_map', 
                                      # cmap=default_cmap_black_white,
                                      show_colorbar=False,
                                      sign='positive',
                                      outlier_perc=args.outlier_perc) # ! default is 10%

    outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_blended_heat_map_pos.jpg', this_image_name)
    image[0].savefig( os.path.join(folder_path, outpath) , bbox_inches='tight', pad_inches=0.0, dpi=400) 

  
    # ! heatmap alone to be converted to blackwhite later? 
    # image = viz.visualize_image_attr( attribution_np_dim_hwc, # ! (224, 224, 3), h x w x chanels
    #                                   original_resize[b].detach().numpy(), # @original_resize is tensor, even when we kept it as numpy, so we have to convert to np
    #                                   method='heat_map', 
    #                                   cmap=default_cmap_black_white,
    #                                   show_colorbar=False,
    #                                   sign='all',
    #                                   outlier_perc=args.outlier_perc) # ! default is 10%

    # outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_heatmapall.jpg', this_image_name)
    # image[0].savefig( os.path.join(folder_path, outpath) , bbox_inches='tight', pad_inches=0.0, dpi=400) 

    
    # ! heatmap alone to be converted to blackwhite later? 
    image = viz.visualize_image_attr( attribution_np_dim_hwc, # ! (224, 224, 3), h x w x chanels
                                      original_resize[b].detach().numpy(), # @original_resize is tensor, even when we kept it as numpy, so we have to convert to np
                                      method='heat_map', 
                                      cmap=default_cmap_black_white,
                                      show_colorbar=False,
                                      sign='positive',
                                      outlier_perc=args.outlier_perc) # ! default is 10%

    outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_heatmappositive.jpg', this_image_name)
    image[0].savefig( os.path.join(folder_path, outpath) , bbox_inches='tight', pad_inches=0.0, dpi=400) 


    # ! heatmap alone to be converted to blackwhite later? 
    # image = viz.visualize_image_attr( attribution_np_dim_hwc, # ! (224, 224, 3), h x w x chanels
    #                                   original_resize[b].detach().numpy(), # @original_resize is tensor, even when we kept it as numpy, so we have to convert to np
    #                                   method='heat_map', 
    #                                   cmap=default_cmap_black_white,
    #                                   show_colorbar=False,
    #                                   sign='negative',
    #                                   outlier_perc=args.outlier_perc) # ! default is 10%

    # outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_heatmapnegative.jpg', this_image_name)
    # image[0].savefig( os.path.join(folder_path, outpath) , bbox_inches='tight', pad_inches=0.0, dpi=400) 

    
    # ! side by side sign contribution
    image = viz.visualize_image_attr_multiple(attribution_np_dim_hwc,
                                              original_resize[b].detach().numpy(),
                                              ["original_image", "heat_map"],
                                              ["all", "all"],
                                              # cmap=default_cmap,
                                              show_colorbar=True, 
                                              outlier_perc=args.outlier_perc)

    outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_bysideSign.jpg', this_image_name)
    image[0].savefig( os.path.join(folder_path, outpath) , bbox_inches='tight', pad_inches=0.0, dpi=400) 


    # ! side by side positive only
    # image = viz.visualize_image_attr_multiple(attribution_np_dim_hwc,
    #                                           original_resize[b].detach().numpy(),
    #                                           ["original_image", "heat_map"],
    #                                           ["all", "positive"],
    #                                           cmap=default_cmap_black_white,
    #                                           show_colorbar=True, 
    #                                           outlier_perc=args.outlier_perc)

    # outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_bysidePositive.jpg', this_image_name)
    # image[0].savefig( os.path.join(folder_path, outpath), bbox_inches='tight', pad_inches=0.0, dpi=400) 
    # print ('save ',os.path.join(folder_path, outpath))


  if args.save_attribution_np: 
    outpath = re.sub ( r'\.(jpg|png|jpeg|PNG)', '_attribution_np.pickle', this_image_name)
    pickle.dump(attribution_np, file=os.path.join(folder_path, outpath))
                    
  # end. 
  return attribution_np


