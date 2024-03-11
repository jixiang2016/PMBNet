import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
import math
cv2.setNumThreads(1) # Avoid deadlock when using DataLoader


class MultiRealDataset(Dataset):
    def __init__(self, data_root, dataset_name, data_mode, dataset_cls, \
                InterNum=0, IntraNum_B0=0,IntraNum_B1=0,sample_type='all',skip_inter=False):
        self.skip_inter = skip_inter
        self.dataset_cls = dataset_cls
        self.data_mode = data_mode
        self.data_root = data_root  # '/media/zhongyi/D/data/RD_VFI'
        self.dataset_name = dataset_name
        self.InterNum = InterNum
        self.IntraNum_B0 = IntraNum_B0
        self.IntraNum_B1 = IntraNum_B1
        self.sample_type = sample_type
        self.image_root = os.path.join(data_root, dataset_cls) # /media/zhongyi/D/data/RD_VFI/train
        self.prepare_data()
    
    def prepare_data(self):
        seqs_list = os.listdir(self.image_root)
        seqs_list = sorted(seqs_list)
        self.sample_paths = []
        for seq_name in seqs_list:
            if self.data_mode == 'Blur':
                seq_rsgr_path = os.path.join(self.image_root,seq_name,'GS','RGB')
            else:
                seq_rsgr_path = os.path.join(self.image_root,seq_name,self.data_mode,'RGB')
            seq_gt_path = os.path.join(self.image_root,seq_name,'HS','RGB')  
            seq_imgs = [img for img in os.listdir(seq_rsgr_path) if self.is_image(img) ]      
            seq_imgs = sorted(seq_imgs,key=lambda x:int(os.path.splitext(x)[0]))        
            seq_imgs_gt = [img for img in os.listdir(seq_gt_path) if self.is_image(img) ]      
            seq_imgs_gt = sorted(seq_imgs_gt,key=lambda x:int(os.path.splitext(x)[0]))   

            for idx in range(0,len(seq_imgs)):
                if (idx+1) > len(seq_imgs)-1: 
                    break
                sample = {}
                B0_path = os.path.join(seq_rsgr_path,seq_imgs[idx])
                B1_path = os.path.join(seq_rsgr_path,seq_imgs[idx+1])
                               
                IntraNum = self.IntraNum_B0+self.IntraNum_B1-1
                all_num = self.InterNum+IntraNum

                gts_B0 = seq_imgs_gt[idx*all_num:idx*all_num+IntraNum][-self.IntraNum_B0:]
                gts_Inter = seq_imgs_gt[idx*all_num+IntraNum:(idx+1)*all_num]
                gts_B1 = seq_imgs_gt[(idx+1)*all_num:(idx+1)*all_num+self.IntraNum_B1]
                
                gts_all_path = gts_B0 + gts_Inter +  gts_B1
                gts_all_path = [os.path.join(seq_gt_path, gts_path) for gts_path in gts_all_path]
                
                gts_all_num = len(gts_all_path)
                S0_path = gts_all_path[0]
                S1_path = gts_all_path[-1]
                
                if self.dataset_cls == 'train' or self.sample_type == 'random':
                    
                    if self.skip_inter ==True:
                        selected_ts = list(range(1, (gts_all_num - 1)))[:self.IntraNum_B0-1]+\
                                     list(range(1, (gts_all_num - 1)))[-(self.IntraNum_B1-1):]
                        interIdx_idx = random.randint(0,(self.IntraNum_B0+self.IntraNum_B1-3) )
                        interIdx = selected_ts[interIdx_idx]
                        
                    else:
                        interIdx = random.randint(1, (gts_all_num - 2))
                     
                    St_path = gts_all_path[interIdx]
                    t_value = interIdx/(gts_all_num-1)  # (0,1)
                    sample.update({
                       'B0_path':B0_path,
                       'B1_path':B1_path,
                       'S0_path':S0_path,
                       'S1_path':S1_path,
                       'St_path':St_path,
                       't_value':t_value,
                    })
                    self.sample_paths.append(sample)
                
                else:
                    sample.update({
                       'B0_path':B0_path,
                       'B1_path':B1_path,
                       'S0_path':S0_path,
                       'S1_path':S1_path,
                    })
                    for interIdx in range(1,gts_all_num-1):
                        
                        if self.skip_inter ==True:
                            selected_ts = list(range(1, (gts_all_num - 1)))[:self.IntraNum_B0-1]+\
                                         list(range(1, (gts_all_num - 1)))[-(self.IntraNum_B1-1):]
                            if interIdx not in selected_ts:
                                continue
                        
                        sample['St_path'] = gts_all_path[interIdx]
                        sample['t_value'] = interIdx/(gts_all_num-1)
                        self.sample_paths.append(sample.copy())
               
    
    def is_image(self, img):
        img_types = ['.PNG','.png','.JPG','.jpg','.JPEG','.jpeg']
        ext_name = os.path.splitext(img)[-1]
        if ext_name in img_types:
            return True
        else:
            return False   
    
    def __len__(self):
        return len(self.sample_paths)
    
    def getimg(self, idx):
        B0_img = cv2.imread(self.sample_paths[idx]['B0_path'])
        B1_img = cv2.imread(self.sample_paths[idx]['B1_path'])
        S0_img = cv2.imread(self.sample_paths[idx]['S0_path'])
        S1_img = cv2.imread(self.sample_paths[idx]['S1_path'])
        St_img = cv2.imread(self.sample_paths[idx]['St_path'])
        t_value = self.sample_paths[idx]['t_value']
        
        t_value = np.expand_dims(np.array(t_value, dtype=np.float32), 0)
        img_arr = np.concatenate([B0_img,B1_img],2)  # (H,W,C*2)
        gt_arr = np.concatenate([S0_img,S1_img,St_img], 2)  ## (H,W,C*3)
        return img_arr, gt_arr, t_value
           
    def crop(self,imgs_arr, gts_arr, h, w):
        ih, iw, _ = imgs_arr.shape
        
        # dorp image boundary before corpping
        #h_offset =16   # offset of height
        #w_offset =2   # offset of width
        
        h_offset =0   # offset of height
        w_offset =0   # offset of width        
        x = np.random.randint(0 + h_offset, ih - h_offset - h + 1)
        y = np.random.randint(0 + w_offset, iw - w_offset - w + 1)
        
        #x = np.random.randint(0, ih - h + 1)
        #y = np.random.randint(0, iw - w + 1)

        imgs_arr = imgs_arr[x:x+h, y:y+w, :]
        gts_arr = gts_arr[x:x+h, y:y+w, :]
        
        return imgs_arr, gts_arr
               
    def __getitem__(self, idx):
        ## img_arr:(H,W,C*2),[B0,B1]   gt_arr:(H,W,C*3),[S0,S1,St]    t_value :(1,)
        imgs_arr, gts_arr, t_value = self.getimg(idx)  

        B0_path = os.path.splitext(self.sample_paths[idx]['B0_path'])[0]
        B1_path = os.path.splitext(self.sample_paths[idx]['B1_path'])[0]
        S0_path = os.path.splitext(self.sample_paths[idx]['S0_path'])[0]
        S1_path = os.path.splitext(self.sample_paths[idx]['S1_path'])[0]
        St_path = os.path.splitext(self.sample_paths[idx]['St_path'])[0]
        B0_id = B0_path.replace(self.image_root,'')
        B1_id = B1_path.replace(self.image_root,'')
        S0_id = S0_path.replace(self.image_root,'')
        S1_id = S1_path.replace(self.image_root,'')
        St_id = St_path.replace(self.image_root,'')
        
        
        if self.dataset_cls == 'train':
            if self.dataset_name== "RD_VFI": 
                imgs_arr, gts_arr = self.crop(imgs_arr, gts_arr, 480, 480) # "RD_VFI": (480,480)  "GOPRO-VFI_copy": (512,512)

            ### augmentation 
            if random.uniform(0, 1) < 0.5 and self.data_mode != 'RSGR': # vertical flipping
                imgs_arr = imgs_arr[::-1]
                gts_arr = gts_arr[::-1]
            
            if random.uniform(0, 1) < 0.5: # horizontal flipping
                imgs_arr = imgs_arr[:, ::-1]
                gts_arr = gts_arr[:, ::-1]
            
            if random.uniform(0, 1) < 0.5: #exchange B0,B1
                imgs_arr = np.concatenate((imgs_arr[:, :, 3:],imgs_arr[:, :, :3]), 2) 
                gts_arr = np.concatenate((gts_arr[:, :, 3:6],gts_arr[:, :, :3], gts_arr[:, :, 6:]), 2)
                t_value = 1.0 - t_value

        imgs_tensor = torch.from_numpy(imgs_arr.copy()).permute(2, 0, 1)  ## (C*2,H,W),[B0,B1]
        gts_tensor = torch.from_numpy(gts_arr.copy()).permute(2, 0, 1) ## (C*3,H,W), [S0,S1,St]
        t_value = torch.from_numpy(t_value)
        batch = { 'img':imgs_tensor , 'label':gts_tensor}
        return batch,t_value, S0_id,S1_id,St_id