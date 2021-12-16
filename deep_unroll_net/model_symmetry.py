import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from package_core.model_base import *
from package_core.losses import *
from package_core.flow_utils import *
from package_core.image_proc import *
from net_symmetry import *

class ModelSymmetry(ModelBase):
    def __init__(self, opts):
        super(ModelSymmetry, self).__init__()
        self.opts = opts

        self.net_G = Net_flow_symmetry(opts.n_chan,
                                        opts.n_init_feat,
                                        est_vel=False,
                                        md=[4,4,4,4]).cuda()

        self.print_networks(self.net_G)

        if self.opts.is_training:
            # create optimizer
            self.optimizer_G = torch.optim.Adam([{'params': self.net_G.parameters()},], lr=opts.lr)            
            self.optimizer_names = ['G']
            self.build_lr_scheduler()

            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            self.loss_fn_tv2 = VariationLoss(nc=2)

            self.downsample2 = nn.AvgPool2d(2, stride=2)

        if not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)

    def set_input(self, _input):
        im_rs, im_gs, gt_flow, cH = _input
        self.im_rs = im_rs.cuda()
        self.im_gs = im_gs
        self.gt_flow = gt_flow
        self.cH = cH

        if self.im_gs is not None:
            self.im_gs = self.im_gs.cuda()
        if self.gt_flow is not None:
            self.gt_flow = self.gt_flow.cuda()

    def forward(self):
        pred_im, pred_mask, pred_flow = self.net_G(self.im_rs)
        return pred_im, pred_mask, pred_flow

    def optimize_parameters(self):
        self.pred_im, self.pred_mask, self.pred_flow = self.forward()
        
        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1 = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_consistency = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_flow_smoothness = torch.tensor([0.], requires_grad=True).cuda().float()

        #===========================================================#
        #                  Prepare ground truth data                #
        #===========================================================#
        im_gs_down1 = self.downsample2(self.im_gs)
        im_gs_down2 = self.downsample2(im_gs_down1)
        im_gs_down3 = self.downsample2(im_gs_down2)
        
        
        self.im_gs_clone=[im_gs_down1]
        self.im_gs_clone.append(im_gs_down2)
        self.im_gs_clone.append(im_gs_down3)

        self.nlvs_ = len(self.pred_im)
        self.nlvs  = len(self.pred_flow)
        
        #===========================================================#
        #                       Compute losses                      #
        #===========================================================#
        for lv in range(self.nlvs):
            if self.pred_flow[lv] is not None and self.opts.lamda_flow_smoothness>1e-6:
                self.loss_flow_smoothness += self.opts.lamda_flow_smoothness *\
                                             self.loss_fn_tv2(self.pred_flow[lv], mean=True)
        
        for lv in range(3):  
            self.loss_perceptual += self.opts.lamda_perceptual *\
                                    self.loss_fn_perceptual.get_loss(self.pred_im[3*lv], self.im_gs_clone[lv])

            self.loss_L1 += self.opts.lamda_L1 *\
                            self.loss_fn_L1(self.pred_im[3*lv], self.im_gs_clone[lv], self.pred_mask[2*lv], mean=True)
        
        #Supervized in the full resolution
        self.loss_perceptual += self.opts.lamda_perceptual *\
                                    self.loss_fn_perceptual.get_loss(self.pred_im[-1], self.im_gs)
        self.loss_L1 += 2.*self.opts.lamda_L1 *\
                            self.loss_fn_L1(self.pred_im[-1], self.im_gs, mean=True)
        
        for lv in range(3):
            self.loss_consistency += self.opts.lamda_consistency *\
                            self.loss_fn_L1(self.pred_im[3*lv+1], self.im_gs_clone[lv], self.pred_mask[2*lv], mean=True)
            self.loss_consistency += self.opts.lamda_consistency *\
                            self.loss_fn_L1(self.pred_im[3*lv+2], self.im_gs_clone[lv], self.pred_mask[2*lv+1], mean=True)
            
        # sum them up
        self.loss_G = self.loss_L1 +\
                        self.loss_perceptual +\
                        self.loss_flow_smoothness +\
                        self.loss_consistency

        # Optimize 
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step() 

    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_G, 'G', label, self.opts.log_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_G, 'G', label, self.opts.log_dir)
        
    def get_current_scalars(self):
        losses = {}
        losses['loss_G'] = self.loss_G.item()
        losses['loss_L1'] = self.loss_L1.item()
        losses['loss_perceptual'] = self.loss_perceptual.item()
        losses['loss_consistency'] = self.loss_consistency.item()
        losses['loss_flow_smoothness'] = self.loss_flow_smoothness.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()

        for lv in range(self.nlvs):
            if self.pred_flow[lv] is not None:
                output_visuals['flow_pred_'+str(lv)] = torch.from_numpy(flow_to_numpy_rgb(self.pred_flow[lv]).transpose(0,3,1,2)).float()/255.
                output_visuals['mask_'+str(lv)] = self.pred_mask[lv].clone().repeat(1,3,1,1)
            
            #if self.pred_im[lv] is None:
                #continue
            #output_visuals['im_gs_'+str(lv)] = self.im_gs_clone[lv]
            #output_visuals['im_gs_pred_'+str(lv)] = self.pred_im[lv]
            #output_visuals['res_im_gs_'+str(lv)] = torch.abs(self.pred_im[lv] - self.im_gs_clone[lv])*5.
            
        return output_visuals


