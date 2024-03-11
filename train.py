import os
import argparse
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import bisect

from dataset.MultiRealDataset import *
from model.trainer import Model
from model.pytorch_msssim import ssim_matlab
from utils.logger import Logger
from utils.timer import (Timer,Epoch_Timer)
from utils.distributed_utils import (broadcast_scalar, is_main_process,reduce_dict, synchronize)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

"""  training parameters """
parser.add_argument('--epoch', default=300, type=int) ##1500
parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
parser.add_argument('--batch_size_val', default=16, type=int, help='minibatch size')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=0 , type=float)  # 1e-3  5e-3
parser.add_argument('--training', default=True,  type=bool)
parser.add_argument('--output_dir', default='./train_log',  type=str, required=True, help='path to save training output')
parser.add_argument('--resume', default=False,  type=bool)
parser.add_argument('--resume_file', default=None,  type=str, help='path to resumed model')
parser.add_argument('--should_log', default=True,  type=bool)
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--world_size', default=4, type=int, help='world size')

""" dataset parameters  """
parser.add_argument('--input_dir', default='/media/zhongyi/D/data',  type=str, required=True, help='path to the input dataset folder')
parser.add_argument('--dataset_name', default='GOPROBase',  type=str, required=True, help='Name of dataset to be used')
parser.add_argument("--InterNum", type=int, default=0, help="number of groundtruth for iterpolated frame between two RSGR frames")
parser.add_argument("--IntraNum_B0", type=int, default=5, help="number of used groundtruth for iterpolated frame within B0")
parser.add_argument("--IntraNum_B1", type=int, default=4, help="number of used groundtruth for iterpolated frame within B1")
parser.add_argument('--data_mode', default='',  type=str, required=True, help='data type used to train:RSGR,RS,Blur')

"""  model parameters  """
parser.add_argument('--deblur_flag', default='mimo',  type=str, help='mimo,mimo+ or single')
parser.add_argument('--merge_flag', default='normal',  type=str, help='deformable or normal')

args = parser.parse_args()

# Gradually reduce the learning rate from 3e-4 to 1e-6 using cosine annealing
def get_learning_rate(step):

    if step < 5000:
        mul = step / 5000.
        return args.learning_rate * mul
    else:
        mul = np.cos((step - 5000) / (args.epoch * args.step_per_epoch - 5000.) * math.pi) * 0.5 + 0.5
        return (args.learning_rate - 1e-6) * mul + 1e-6
    

def _summarize_report(prefix="", should_print=True, extra={},log_writer=None,current_iteration=0,max_iterations=0):
        if not is_main_process():
            return
        if not should_print:
            return
        print_str = []
        if len(prefix):
            print_str += [prefix + ":"]
        print_str += ["{}/{}".format(current_iteration, max_iterations)]
        print_str += ["{}: {}".format(key, value) for key, value in extra.items()]
        log_writer.write(','.join(print_str)) 


def train(model):
    log_writer = Logger(args) 
    log_writer.write("Torch version is: " + torch.__version__)
    log_writer.write("===== Model =====")
    log_writer.write(model.net_model)
       
    if is_main_process():
        writer = SummaryWriter('./tensorboard_log/train')
        writer_val = SummaryWriter('./tensorboard_log/validate')
    else:
        writer = None
        writer_val = None


    if args.dataset_name == 'RD_VFI':
        args.InterNum = 0
        args.IntraNum_B0 = 6
        args.IntraNum_B1 = 5
        data_root = os.path.join(args.input_dir, args.dataset_name)
        data_train = MultiRealDataset(data_root=data_root,\
                                dataset_name = args.dataset_name, \
                                data_mode = args.data_mode, \
                                dataset_cls='train',\
                                InterNum=args.InterNum,\
                                IntraNum_B0 = args.IntraNum_B0,\
                                IntraNum_B1 = args.IntraNum_B1)
        dataset_val = MultiRealDataset(data_root=data_root,\
                                dataset_name = args.dataset_name, \
                                data_mode = args.data_mode, \
                                dataset_cls='validate',\
                                InterNum=args.InterNum,\
                                IntraNum_B0 = args.IntraNum_B0,\
                                IntraNum_B1 = args.IntraNum_B1,
                                sample_type='random')
                                
    elif args.dataset_name == 'GOPRO-VFI_copy':
        args.InterNum = 0
        args.IntraNum_B0 = 5
        args.IntraNum_B1 = 4
        data_root = os.path.join(args.input_dir, args.dataset_name)
        data_train = MultiRealDataset(data_root=data_root,\
                                dataset_name = args.dataset_name, \
                                data_mode = args.data_mode, \
                                dataset_cls='train',\
                                InterNum=args.InterNum,\
                                IntraNum_B0 = args.IntraNum_B0,\
                                IntraNum_B1 = args.IntraNum_B1)
        dataset_val = MultiRealDataset(data_root=data_root,\
                                dataset_name = args.dataset_name, \
                                data_mode = args.data_mode, \
                                dataset_cls='test',\
                                InterNum=args.InterNum,\
                                IntraNum_B0 = args.IntraNum_B0,\
                                IntraNum_B1 = args.IntraNum_B1,
                                sample_type='random')
    else:
        raise Exception('not supported dataset!')    
    
    
    sampler = DistributedSampler(data_train)
    train_data = DataLoader(data_train, batch_size=args.batch_size, num_workers=8,\
                            pin_memory=False, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__() # total number of steps per epoch
    val_data = DataLoader(dataset_val, batch_size=args.batch_size_val, pin_memory=False, num_workers=8,shuffle=False)
    

    #resume 
    if args.resume is True:
        log_writer.write("Restore traing from saved model")
        if args.resume_file is None:
            dir_name = args.dataset_name+'_'+args.data_mode
            checkpoint_path = os.path.join(args.output_dir,dir_name,'best.ckpt')
        else:
            checkpoint_path = args.resume_file
        checkpoint_info = model.load_model(path=checkpoint_path)
    
    
    if torch.device("cuda") == device:
        rank = args.local_rank if args.local_rank >=0 else 0
        device_info = "CUDA Device {} is: {}".format(rank, torch.cuda.get_device_name(args.local_rank))
        log_writer.write(device_info, log_all=True)
    log_writer.write("Starting training...")
    log_writer.write("Each epoch includes {} iterations".format(args.step_per_epoch))
    
    
    train_timer = Timer()
    snapshot_timer = Timer()
    max_step = args.step_per_epoch*args.epoch
    

    if args.resume is True:
        step = checkpoint_info['best_monitored_iteration'] + 1
        start_epoch = checkpoint_info['best_monitored_epoch']
        best_dict = checkpoint_info
    else:    
        step = 0 # total training steps across all epochs
        start_epoch = 0
        best_dict={
            'best_monitored_value': 0,
            'best_psnr':0,
            'best_ssim':0,
            'best_monitored_iteration':-1,
            'best_monitored_epoch':-1, 
            'best_monitored_epoch_step':-1,
        }
    
    
    epoch_timer = Epoch_Timer('m')
    for epoch in range(start_epoch,args.epoch):
        sampler.set_epoch(epoch) ## To shuffle data
        if step > max_step:
            break
            
        epoch_timer.tic()
        
        for trainIndex, all_data in enumerate(train_data): 
            learning_rate = get_learning_rate(step)
            
            data = all_data[0]
            for k in data:
                data[k] = data[k].to(device, non_blocking=True) / 255. # Normalize to (0,1), BGR
                data[k].requires_grad = False
                

            ### data['img'] : [B0,B1],(B,C*2,H,W)     data['label']:[S0,S1,St],(B,C*3,H,W)
            #[B0,B1]
            input_frames = data['img']## (B,C*2,H,W)
            # St
            frameT = data['label'][:,-3:]
            ## [S0, S1] 
            input_frames_GT = data['label'][:,:-3]## (B,C*2,H,W)
            ## (B,1)
            t_value = all_data[1].to(device, non_blocking=True)
          
            ## S0S1: (B,6,H,W)  St:(B,3,H,W)
            S0S1,St_pre,other_outputs,info = model.update(\
                                                      input_frames, t_value, frameT,\
                                                      input_frames_GT,learning_rate,training=True)                               
            pred_S0 = S0S1[:,:3]# (B, C, H, W)   
            pred_S1 = S0S1[:,3:]# (B, C, H, W)   
            pred_St = St_pre# (B, C, H, W) 
            pred = torch.cat([pred_S0,pred_S1,pred_St], dim=0) ## (B*out_num, C, H, W)
            
            gt_S0 = input_frames_GT[:,:3]
            gt_S1 = input_frames_GT[:,3:]
            gts_tensor = torch.cat([gt_S0,gt_S1,frameT], dim=0) ## (B*out_num, C, H, W)
      
            MAX_DIFF = 1
            mse = (gts_tensor - pred) * (gts_tensor - pred)
            mse = torch.mean(torch.mean(torch.mean(mse,-1),-1),-1).detach().cpu().data ###(batch*output_num,)
            psnr_aa = 10* torch.log10( MAX_DIFF**2 / mse ) ###(batch*output_num,)
            psnr = torch.mean(psnr_aa)
            
            ssim = ssim_matlab(gts_tensor,pred).detach().cpu().numpy()

            ##### write summary to tensorboard
            if is_main_process():
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/content', info['loss_content'], step)
                writer.add_scalar('loss/total', info['loss_total'], step)
                writer.add_scalar('psnr', psnr, step)  
                writer.add_scalar('ssim', float(ssim), step)  
            
            #Log traing info to screen and file
            should_print = (step % 2000 == 0 and step !=0)
            extra = {}
            if should_print is True:   
                extra.update(
                    {
                        "lr": "{:.2e}".format(learning_rate),
                        "time": train_timer.get_time_since_start(),
                        "train/total_loss":format(info['loss_total'].detach().cpu().numpy(), '.4f' ),
                        "train/loss_content":format(info['loss_content'].detach().cpu().numpy(),'.4f'),
                        "train/psnr":format(psnr,'.4f'),
                        "train/ssim":format(ssim,'.4f'),
                    }
                )
                train_timer.reset()
                val_infor = evaluate(model, val_data, step,writer_val,True)
                extra.update(val_infor)
            _summarize_report(
                                should_print=should_print,
                                extra=extra,
                                prefix=args.dataset_name+'_'+args.data_mode,
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )
                             
                             
            #### Conduct full evaluation and save checkpoint
            if step % 7000 == 0 and step !=0:
                log_writer.write("Evaluation time. Running on full validation set...")
                all_val_infor = evaluate(model, val_data, step,writer_val,False,use_tqdm=True)
                val_extra = {"validation time":snapshot_timer.get_time_since_start()}
                
                if (all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2 > best_dict['best_monitored_value']:
                    best_dict['best_monitored_iteration'] = step    
                    best_dict['best_monitored_epoch_step'] = trainIndex
                    best_dict['best_monitored_epoch'] = epoch
                    best_dict['best_monitored_value'] = float(format((all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2,'.4f'))
                    best_dict['best_ssim'] = all_val_infor['val/ssim']
                    best_dict['best_psnr'] =all_val_infor['val/psnr']
                    model.save_model(args,step,best_dict, update_best=True) 
                else:
                    model.save_model(args,step,best_dict, update_best=False) 
                
                val_extra.update(
                    {'current_psnr':all_val_infor['val/psnr'],
                     'current_ssim':all_val_infor['val/ssim'],
                    }
                )
                val_extra.update(best_dict)
                prefix = "{}: full val".format(args.dataset_name+'_'+args.data_mode) 
                _summarize_report(
                                extra=val_extra,
                                prefix=prefix,
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )
                snapshot_timer.reset()
                gc.collect() # clear up memory
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()
                
            step += 1
            if step > max_step:
                break

        if is_main_process():
            print("EPOCH: %02d    Elapsed time: %4.2f " % (epoch+1, epoch_timer.toc()))
        dist.barrier()

def evaluate(model, val_data, step,writer_val,single_batch,use_tqdm=False):

    psnr_list = []
    ssim_list = []
    disable_tqdm = not use_tqdm
   
    for testIndex, all_data in enumerate(tqdm(val_data,disable=disable_tqdm)):
        data = all_data[0]
        for k in data:
            data[k] = data[k].to(device, non_blocking=True) / 255.      #### BGR [0,1]
            data[k].requires_grad = False
        ### data['img']: [B0,B1],(B,C*2,H,W)    data['label']: [S0,S1,St] (B,C*3,H,W)
        ### [B0,B1]
        input_frames = data['img']
        t_value = all_data[1].to(device, non_blocking=True)
        # St
        gt_St = data['label'][:,-3:]  ## (B,C,H,W)
        ## [S0,S1]
        S0S1_GT_frames = data['label'][:,:-3] ## (B,C,H,W)

        with torch.no_grad():
            ## S0S1: (B,6,H,W)  St:(B,3,H,W)
            S0S1,St_pre,other_outputs,info = model.update(\
                                                      input_frames, t_value, gt_St,\
                                                      S0S1_GT_frames,training=False)

        pred_S0 = S0S1[:,:3].clamp(0,1)# (B, C, H, W)   
        pred_S1 = S0S1[:,3:].clamp(0,1)# (B, C, H, W)   
        pred_St = St_pre.clamp(0,1)# (B, C, H, W) 
        pred = torch.cat([pred_S0,pred_S1,pred_St], dim=0) ## (B*out_num, C, H, W)
            
        gt_S0 = S0S1_GT_frames[:, :3]  # [B,C,H,W]
        gt_S1 = S0S1_GT_frames[:, 3:]  # [B,C,H,W]
        
        gts_tensor = torch.cat([gt_S0,gt_S1,gt_St], dim=0) ## (B*out_num, C, H, W)

        MAX_DIFF = 1 ## because data is normalized into (0,1),so max difference is 1
        mse = (gts_tensor - pred) * (gts_tensor - pred)
        mse = torch.mean(torch.mean(torch.mean(mse,-1),-1),-1).detach().cpu().data ###(batch*output_num,)
        psnr_aa = 10* torch.log10( MAX_DIFF**2 / mse ) ###(batch*output_num,)
        psnr = torch.mean(psnr_aa)
        psnr_list.append(psnr)
        
        ssim = ssim_matlab(gts_tensor,pred).detach().cpu().numpy()
        ssim_list.append(ssim)
        
        if single_batch is True:
            break
        
    if is_main_process() and single_batch is False:
       writer_val.add_scalar('psnr', np.array(psnr_list).mean(), step)
       writer_val.add_scalar('ssim', np.array(ssim_list).mean(), step)

    return {
            'val/ssim': float(format(np.mean(ssim_list),'.4f')),
            'val/psnr': float(format(np.mean(psnr_list),'.4f')),
            }



if __name__ == "__main__":    
    
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank) 
    
    # For reproduction 
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # To accelerate training process when network structure and inputsize are fixed
    torch.backends.cudnn.benchmark = True
    
    model = Model(config=args,local_rank=args.local_rank)
    train(model)
        
