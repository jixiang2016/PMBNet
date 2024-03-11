import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import torch.nn.init as init
import itertools
from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import datetime
import os
import math

from utils.distributed_utils import (broadcast_scalar, is_main_process,reduce_dict, synchronize)
from model.loss import *
from model.PMBNet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self,config,local_rank=-1):
    
        self.local_rank = local_rank
        self.deblur_flag = config.deblur_flag
        self.training = config.training
        self.net_model = PMBNet(config)
        self.net_model.to(device)
        if config.training:
        
            self.optimG = torch.optim.Adam(self.net_model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), weight_decay=config.weight_decay)
            self.l1_loss = torch.nn.L1Loss()  # L1 loss 
            
        if local_rank != -1:
            self.net_model = DDP(self.net_model, device_ids=[local_rank], find_unused_parameters=True,output_device=local_rank)
        
        
    def load_model(self, path):
        if device == "cuda":
            ckpt = torch.load(path, map_location=device)
        else:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        ckpt_model = ckpt["model"]
        
        new_dict = {}
        
        for attr in ckpt_model:
            if self.local_rank==-1 and attr.startswith("module."): # non-parallel mode load model trained by parallel mode
                new_dict[attr.replace("module.", "", 1)] = ckpt_model[attr]
            elif self.local_rank >=0 and not attr.startswith("module."):# parallel mode load model trained by non-parallel mode
                new_dict["module." + attr] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]             
                
        self.net_model.load_state_dict(new_dict)        
        
        return {
            'best_monitored_value': ckpt['best_monitored_value'],
            'best_psnr':ckpt['best_psnr'],
            'best_ssim':ckpt['best_ssim'],
            'best_monitored_iteration':ckpt['best_monitored_iteration'],
            'best_monitored_epoch':ckpt['best_monitored_epoch'], 
            'best_monitored_epoch_step':ckpt['best_monitored_epoch_step'],
        }      
        
    def inference(self, input_frames, t_value):
        self.net_model.eval() 
        #### Padding, todo
        ## Make sure input(height,width) is multiples of (32,32)
       
        ## St: (B,3,H,W)
        flow_list, corrected_img_list, St, flow_t1, flow_t0 = self.net_model(input_frames, t_value, training=False )
        other_outputs = {'flow_list':flow_list, 'flow_t1':flow_t1,'flow_t0':flow_t0}
        return corrected_img_list[-1][-1], St, other_outputs
           
    def update(self, input_frames, t_value, frameT, input_frames_GT, \
               learning_rate=0, training=False):
        
        # input_frames : (B,C*2,H,W)  [B0, B1]        t_value: (B,1)
        # frameT: (B,C,H,W)   input_frames_GT: (B,C*2,H,W) [S0, S1]
        if training:
            for param_group in self.optimG.param_groups:
                param_group['lr'] = learning_rate 
            self.net_model.train()
        else:
            self.net_model.eval()  

        if training:
            flow_list, corrected_img_list, St, flow_t1, flow_t0 = self.net_model(input_frames, t_value, training )
            other_outputs = {'flow_list':flow_list, 'flow_t1':flow_t1,'flow_t0':flow_t0}
        else:
            #### Padding,  todo
            ## Make sure input(height,width) is multiples of (32,32)
            flow_list, corrected_img_list, St,flow_t1, flow_t0= self.net_model(input_frames, t_value, training )
            other_outputs = {'flow_list':flow_list, 'flow_t1':flow_t1,'flow_t0':flow_t0}
            
        input_frames_GT_2 = F.interpolate(input_frames_GT, scale_factor = 1./2 , mode="bilinear", align_corners=False)
        input_frames_GT_4 = F.interpolate(input_frames_GT_2, scale_factor = 1./2 , mode="bilinear", align_corners=False)

        if self.deblur_flag =='mimo' or self.deblur_flag =='mimo+':
            input_frames_GT_4_2 = F.interpolate(input_frames_GT_4, scale_factor = 1./2 , mode="bilinear", align_corners=False)
            input_frames_GT_4_4 = F.interpolate(input_frames_GT_4_2, scale_factor = 1./2 , mode="bilinear", align_corners=False)
            loss_level4 = (self.l1_loss(corrected_img_list[0][0][:,:3], input_frames_GT_4_4[:,:3]) +\
                          self.l1_loss(corrected_img_list[0][0][:,3:], input_frames_GT_4_4[:,3:]) +\
                          self.l1_loss(corrected_img_list[0][1][:,:3], input_frames_GT_4_2[:,:3]) +\
                          self.l1_loss(corrected_img_list[0][1][:,3:], input_frames_GT_4_2[:,3:]) +\
                          self.l1_loss(corrected_img_list[0][2][:,:3], input_frames_GT_4[:,:3]) +\
                          self.l1_loss(corrected_img_list[0][2][:,3:], input_frames_GT_4[:,3:]))/3.0
            loss_level2 = (self.l1_loss(corrected_img_list[1][0][:,:3], input_frames_GT_4_2[:,:3]) +\
                          self.l1_loss(corrected_img_list[1][0][:,3:], input_frames_GT_4_2[:,3:]) +\
                          self.l1_loss(corrected_img_list[1][1][:,:3], input_frames_GT_4[:,:3]) +\
                          self.l1_loss(corrected_img_list[1][1][:,3:], input_frames_GT_4[:,3:]) +\
                          self.l1_loss(corrected_img_list[1][2][:,:3], input_frames_GT_2[:,:3]) +\
                          self.l1_loss(corrected_img_list[1][2][:,3:], input_frames_GT_2[:,3:]))/3.0
            loss_level1 = (self.l1_loss(corrected_img_list[2][0][:,:3], input_frames_GT_4[:,:3]) +\
                          self.l1_loss(corrected_img_list[2][0][:,3:], input_frames_GT_4[:,3:]) +\
                          self.l1_loss(corrected_img_list[2][1][:,:3], input_frames_GT_2[:,:3]) +\
                          self.l1_loss(corrected_img_list[2][1][:,3:], input_frames_GT_2[:,3:]) +\
                          self.l1_loss(corrected_img_list[2][2][:,:3], input_frames_GT[:,:3]) +\
                          self.l1_loss(corrected_img_list[2][2][:,3:], input_frames_GT[:,3:]))/3.0
        else: 
            loss_level4 = self.l1_loss(corrected_img_list[0][0][:,:3], input_frames_GT_4[:,:3]) +\
                          self.l1_loss(corrected_img_list[0][0][:,3:], input_frames_GT_4[:,3:])
            loss_level2 = self.l1_loss(corrected_img_list[1][0][:,:3], input_frames_GT_2[:,:3]) +\
                          self.l1_loss(corrected_img_list[1][0][:,3:], input_frames_GT_2[:,3:])
            loss_level1 = self.l1_loss(corrected_img_list[2][0][:,:3], input_frames_GT[:,:3]) +\
                          self.l1_loss(corrected_img_list[2][0][:,3:], input_frames_GT[:,3:])

        loss_s0_s1 = (loss_level1+loss_level2+loss_level1)/3.0         
        loss_st = self.l1_loss(St, frameT)
        loss_G =  loss_st + loss_s0_s1

        if training:
            self.optimG.zero_grad()
            loss_G.backward()
            #torch.nn.utils.clip_grad_norm_( self.net_model.parameters() , 50 ) 
            self.optimG.step()       
        return corrected_img_list[-1][-1], St, other_outputs, {
            'loss_content': loss_G,
            'loss_st': loss_st,
            'loss_s0_s1': loss_s0_s1,
            'loss_total': loss_G,
            }
    
    def save_model(self, args,step,best_dict,update_best):
    
        if not is_main_process():
            return
        
        dir_name = args.dataset_name+'_'+args.data_mode
        dir_path = os.path.join(args.output_dir,dir_name,'models')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        ckpt_filepath = os.path.join(
            dir_path, "model_%d.ckpt" % step
        )
        best_ckpt_filepath = os.path.join(
            args.output_dir,dir_name, "best.ckpt"
        )
        ckpt = {
            "model": self.net_model.state_dict(),
           
        }
        
        ckpt.update(best_dict)
        torch.save(ckpt, ckpt_filepath)
        
        if update_best:
            torch.save(ckpt, best_ckpt_filepath)
    
    

