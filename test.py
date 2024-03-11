import os
import sys
import cv2
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.trainer import Model
from dataset.MultiRealDataset import *
from lpips import lpips
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)

parser = argparse.ArgumentParser()

'''testing parameters'''
parser.add_argument('--training', default=False,  type=bool)
parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
parser.add_argument('--output_dir', default='',  type=str, required=True, help='path to save testing output')
parser.add_argument('--model_dir', default="./train_log/GOPROBase_RSGR_3_5/best.ckpt",  
                    type=str, help='path to the pretrained model folder')
parser.add_argument('--keep_frames', action='store_true', default=False, help='save interpolated frames')
parser.add_argument('--keep_flows', action='store_true', default=False, help='save predicted flows')
parser.add_argument('--skip_inter',type=bool, default=False)

'''dataset parameters'''
parser.add_argument('--input_dir', default='/media/zhongyi/D/data/GOPRO_RSGR',  type=str, required=True, help='path to the input dataset folder')
parser.add_argument('--dataset_name', default='GOPROBase',  type=str, required=True, help='Name of dataset to be used')

parser.add_argument("--InterNum", type=int, default=0, help="number of groundtruth for iterpolated frame between two RSGR frames")
parser.add_argument("--IntraNum_B0", type=int, default=5, help="number of used groundtruth for iterpolated frame within B0")
parser.add_argument("--IntraNum_B1", type=int, default=4, help="number of used groundtruth for iterpolated frame within B1")
parser.add_argument('--data_mode', default='',  type=str, required=True, help='data type used to train:RSGR,RS,Blur')

'''model parameter '''
parser.add_argument('--deblur_flag', default='mimo',  type=str, help='mimo,mimo+ or single')
parser.add_argument('--merge_flag', default='normal',  type=str, help='deformable or normal')

args = parser.parse_args()

def test(model): 
    model.load_model(path=args.model_dir)
 
    if args.dataset_name == 'RD_VFI':
        args.InterNum = 0
        args.IntraNum_B0 = 6
        args.IntraNum_B1 = 5
        data_root = os.path.join(args.input_dir, args.dataset_name)
        dataset_val = MultiRealDataset(data_root=data_root,\
                                dataset_name = args.dataset_name, \
                                data_mode = args.data_mode, \
                                dataset_cls='test',\
                                InterNum=args.InterNum,\
                                IntraNum_B0 = args.IntraNum_B0,\
                                IntraNum_B1 = args.IntraNum_B1) # skip_inter = args.skip_inter for only testing on intra frames
    
    elif args.dataset_name == 'GOPRO-VFI_copy':
        args.InterNum = 0
        args.IntraNum_B0 = 5
        args.IntraNum_B1 = 4
        data_root = os.path.join(args.input_dir, args.dataset_name)
        dataset_val = MultiRealDataset(data_root=data_root,\
                                dataset_name = args.dataset_name, \
                                data_mode = args.data_mode, \
                                dataset_cls='test',\
                                InterNum=args.InterNum,\
                                IntraNum_B0 = args.IntraNum_B0,\
                                IntraNum_B1 = args.IntraNum_B1)
    
    else:
        raise Exception('not supported dataset!') 

    val_data = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=8,shuffle=False)
    
    psnr_list = []
    psnr_dict = {}
    psnr_time = {}
        
    ssim_list = []
    ssim_dict = {}
    ssim_time = {}
    
    lpips_list = []
    lpips_dict = {}
    lpips_time ={}
    
    psnr_vfi_list = []
    ssim_vfi_list = []
    lpips_vfi_list = []
    
    psnr_corr_list = []
    ssim_corr_list = []
    lpips_corr_list = []
    

    if args.skip_inter == True:
        constantNum = args.IntraNum_B0+ args.IntraNum_B1-2
    else:
        constantNum = args.IntraNum_B0+ args.IntraNum_B1+args.InterNum-2
    countNum = 1
    
    for textIndex, all_data in enumerate(tqdm(val_data)):

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
        S0S1_GT_frames = data['label'][:,:-3]
        S0_Path, S1_Path, St_Path = all_data[2],all_data[3],all_data[4]  # '/GOPR0384_11_00/GT/000001/000004' or '/_Scene12_2/HS/RGB/000005'
        
        with torch.no_grad():     
            ## S0S1: (B,6,H,W)  St:(B,3,H,W)
            S0S1,St_pre,other_outputs = model.inference(input_frames, t_value)
        
        
        pred_S0 = S0S1[:,:3].clamp(0,1)# (B, C, H, W)   
        pred_S1 = S0S1[:,3:].clamp(0,1)# (B, C, H, W)   
        pred_St = St_pre.clamp(0,1)# (B, C, H, W) 
        gt_S0 = S0S1_GT_frames[:, :3]  # [B,C,H,W]
        gt_S1 = S0S1_GT_frames[:, 3:]  # [B,C,H,W]

        
        batch_size = gt_St.shape[0]
        for b_id in range(batch_size):     
            p_S0 = pred_S0[b_id] # (3,h,w)
            p_S1 = pred_S1[b_id]
            p_St = pred_St[b_id]
            g_S0 = gt_S0[b_id]  # (3,c,h)
            g_S1 = gt_S1[b_id]
            g_St = gt_St[b_id]
            g_S0_id = S0_Path[b_id]  
            g_S1_id = S1_Path[b_id] 
            g_St_id = St_Path[b_id] 
            tt = t_value[b_id]
            
            seq_name = g_S0_id.split('/')[1]  
            S0_img_name = g_S0_id.split('/')[3]  
            S0_img_gt = g_S0_id.split('/')[-1]    
            S1_img_name = g_S1_id.split('/')[3]  
            S1_img_gt = g_S1_id.split('/')[-1]    
            St_img_name = g_St_id.split('/')[3]   
            St_img_gt = g_St_id.split('/')[-1]    

            S0_save_path = args.output_dir+'/'+args.dataset_name+'/'+seq_name+'/'+S0_img_name
            S1_save_path = args.output_dir+'/'+args.dataset_name+'/'+seq_name+'/'+S1_img_name
            St_save_path = args.output_dir+'/'+args.dataset_name+'/'+seq_name+'/'+St_img_name
            
            if args.keep_frames is True:
                if not os.path.exists(S0_save_path):
                    os.makedirs(S0_save_path, exist_ok=True)
                if not os.path.exists(S1_save_path):
                    os.makedirs(S1_save_path, exist_ok=True)
                if not os.path.exists(St_save_path):
                    os.makedirs(St_save_path, exist_ok=True)

            S0_ssim = ssim_matlab(g_S0.unsqueeze(0),p_S0.unsqueeze(0)).cpu().numpy()
            S1_ssim = ssim_matlab(g_S1.unsqueeze(0),p_S1.unsqueeze(0)).cpu().numpy()
            St_ssim = ssim_matlab(g_St.unsqueeze(0),p_St.unsqueeze(0)).cpu().numpy()
            
            MAX_DIFF = 1 ## because data is normalized into (0,1),so max difference is 1
            S0_psnr = 10* math.log10( MAX_DIFF**2 / (torch.mean((g_S0 - p_S0) * (g_S0 - p_S0)).cpu().data) )
            S1_psnr = 10* math.log10( MAX_DIFF**2 / (torch.mean((g_S1 - p_S1) * (g_S1 - p_S1)).cpu().data) )
            St_psnr = 10* math.log10( MAX_DIFF**2 / (torch.mean((g_St - p_St) * (g_St - p_St)).cpu().data) )
            
            
            S0_lpips=loss_fn_alex(p_S0, g_S0).cpu().item() # compute LPIPS
            S1_lpips=loss_fn_alex(p_S1, g_S1).cpu().item()
            St_lpips=loss_fn_alex(p_St, g_St).cpu().item()
            
            
            if args.keep_frames is True:
                p_S0_s = (p_S0.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                cv2.imwrite(os.path.join(S0_save_path,S0_img_gt+'.png'),p_S0_s)
                p_S1_s = (p_S1.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                cv2.imwrite(os.path.join(S1_save_path,S1_img_gt+'.png'),p_S1_s)
                p_St_s = (p_St.permute(1,2,0).cpu().numpy() * 255).astype('uint8')
                cv2.imwrite(os.path.join(St_save_path,St_img_gt+'.png'),p_St_s)


            if countNum%constantNum ==0:
                if '000000' not in psnr_time:
                    psnr_time['000000'] = []
                psnr_time['000000'].append(S0_psnr)
                if '000001' not in psnr_time:
                    psnr_time['000001'] = []
                psnr_time['000001'].append(S1_psnr)
            t_str =  format(tt.cpu().item(),'.5f') 
            if t_str not in psnr_time:
                psnr_time[t_str] = []
            psnr_time[t_str].append(St_psnr)
            if countNum%constantNum==0:
                if '000000' not in ssim_time:
                    ssim_time['000000'] = []
                ssim_time['000000'].append(S0_ssim)
                if '000001' not in ssim_time:
                    ssim_time['000001'] = []
                ssim_time['000001'].append(S1_ssim)
            if t_str not in ssim_time:
                ssim_time[t_str] = []
            ssim_time[t_str].append(St_ssim)
            if countNum%constantNum==0:
                if '000000' not in lpips_time:
                    lpips_time['000000'] = []
                lpips_time['000000'].append(S0_lpips)
                if '000001' not in lpips_time:
                    lpips_time['000001'] = []
                lpips_time['000001'].append(S1_lpips) 
            if t_str not in lpips_time:
                lpips_time[t_str] = []
            lpips_time[t_str].append(St_lpips)

            
            if seq_name not in psnr_dict:
                psnr_dict[seq_name]={}
            if S0_img_name not in psnr_dict[seq_name]:
                psnr_dict[seq_name][S0_img_name] = {}
            psnr_dict[seq_name][S0_img_name][S0_img_gt] = format(S0_psnr,'.4f') # float(format(S0_psnr,'.4f'))
            if S1_img_name not in psnr_dict[seq_name]:
                psnr_dict[seq_name][S1_img_name] = {}
            psnr_dict[seq_name][S1_img_name][S1_img_gt] = format(S1_psnr,'.4f')
            if St_img_name not in psnr_dict[seq_name]:
                psnr_dict[seq_name][St_img_name] = {}
            psnr_dict[seq_name][St_img_name][St_img_gt] = format(St_psnr,'.4f')
            psnr_list.append(St_psnr)
            psnr_vfi_list.append(St_psnr)
            if countNum%constantNum==0:
                psnr_list.append(S0_psnr)
                #psnr_list.append(S1_psnr)
                psnr_corr_list.append(S0_psnr)
            
            if seq_name not in ssim_dict:
                ssim_dict[seq_name]={}
            if S0_img_name not in ssim_dict[seq_name]:
                ssim_dict[seq_name][S0_img_name] = {}
            ssim_dict[seq_name][S0_img_name][S0_img_gt] = format(S0_ssim,'.4f')
            if S1_img_name not in ssim_dict[seq_name]:
                ssim_dict[seq_name][S1_img_name] = {}
            ssim_dict[seq_name][S1_img_name][S1_img_gt] = format(S1_ssim,'.4f')
            if St_img_name not in ssim_dict[seq_name]:
                ssim_dict[seq_name][St_img_name] = {}
            ssim_dict[seq_name][St_img_name][St_img_gt] = format(St_ssim,'.4f')
            ssim_list.append(St_ssim)
            ssim_vfi_list.append(St_ssim)
            if countNum%constantNum == 0:
                ssim_list.append(S0_ssim)
                #ssim_list.append(S1_ssim)
                ssim_corr_list.append(S0_ssim)
            
            if seq_name not in lpips_dict:
                lpips_dict[seq_name]={}
            if S0_img_name not in lpips_dict[seq_name]:
                lpips_dict[seq_name][S0_img_name] = {}
            lpips_dict[seq_name][S0_img_name][S0_img_gt] = format(S0_lpips,'.4f') # float(format(S0_psnr,'.4f'))
            if S1_img_name not in lpips_dict[seq_name]:
                lpips_dict[seq_name][S1_img_name] = {}
            lpips_dict[seq_name][S1_img_name][S1_img_gt] = format(S1_lpips,'.4f')
            if St_img_name not in lpips_dict[seq_name]:
                lpips_dict[seq_name][St_img_name] = {}
            lpips_dict[seq_name][St_img_name][St_img_gt] = format(St_lpips,'.4f')
            lpips_list.append(St_lpips)
            lpips_vfi_list.append(St_lpips)
            if countNum%constantNum==0:
                lpips_list.append(S0_lpips)
                #lpips_list.append(S1_lpips)
                lpips_corr_list.append(S0_lpips)
            
            countNum += 1
            

    save_dir = args.output_dir+'/'+args.dataset_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # Keep txt record
    for seq_name,img_dict in psnr_dict.items():
        with open(save_dir+'/'+seq_name+'.txt','w') as f:
            for img_name,gt_dict in sorted(img_dict.items(),key=lambda x:x[0]):
                gt_dict = sorted(gt_dict.items(),key=lambda x:x[0])
                gt_dict_ssim = sorted(ssim_dict[seq_name][img_name].items(),key=lambda x:x[0])
                gt_dict_lpips = sorted(lpips_dict[seq_name][img_name].items(),key=lambda x:x[0])
                for xx, yy, zz in zip(gt_dict, gt_dict_ssim,gt_dict_lpips):
                    assert xx[0] == yy[0] == zz[0]
                    f.write( xx[0]+'\t'+ xx[1]+'\t'+ yy[1]+'\t'+zz[1]+'\n')

    with open(save_dir+'/overall_metrics.txt','w') as f:
        f.write('Overall PSNR: %.4f\n'%(np.mean(psnr_list)))
        f.write('Overall SSIM: %.4f\n'%(np.mean(ssim_list)))
        f.write('Overall LPIPS: %.4f\n'%(np.mean(lpips_list)))
        f.write('Correction PSNR: %.4f\n'%(np.mean(psnr_corr_list)))
        f.write('Correction SSIM: %.4f\n'%(np.mean(ssim_corr_list)))
        f.write('Correction LPIPS: %.4f\n'%(np.mean(lpips_corr_list)))
        f.write('Interpolation PSNR: %.4f\n'%(np.mean(psnr_vfi_list)))
        f.write('Interpolation SSIM: %.4f\n'%(np.mean(ssim_vfi_list)))
        f.write('Interpolation LPIPS: %.4f\n'%(np.mean(lpips_vfi_list)))
        
        f.write('metrics by time stamp:\n')
        psnr_time = sorted(psnr_time.items(),key=lambda x:float(x[0]))
        for kk in psnr_time:
            avg_psnr = format(np.mean(kk[1]),'.4f')
            avg_ssim =  format(np.mean(ssim_time[kk[0]]),'.4f') 
            avg_lpips =  format(np.mean(lpips_time[kk[0]]),'.4f')
            f.write( 'tiemstamp:'+kk[0]+'\t'+ avg_psnr+'\t'+ avg_ssim+'\t'+avg_lpips+'\n')
        
        
    print('---------------------------------------------------------------')
    print('Overall PSNR: %.4f'%(np.mean(psnr_list)))
    print('Overall SSIM: %.4f'%(np.mean(ssim_list)))
    print('Overall LPIPS: %.4f'%(np.mean(lpips_list)))
    print('---------------------------------------------------------------')
    


if __name__ == "__main__":    
       
    # For reproduction 
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = Model(config=args)
    
    test(model)
        




