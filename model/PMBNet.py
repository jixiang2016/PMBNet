import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.warplayer import warp

from model.deformable_attention import *



'''
 basic building blocks
'''
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel)) #torch.nn.LayerNorm
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = BasicConv(in_planes, out_planes, 3, stride)
        self.conv2 = BasicConv(out_planes, out_planes, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
        
        
#######################################################

''' proposed architecture PMB-net  '''


class PMBNet(nn.Module):

    def __init__(self, args):
        super(PMBNet, self).__init__()
        
        self.deblur_flag = args.deblur_flag  ## 'mimo', 'single'
        self.merge_flag = args.merge_flag
        self.block0 = PMBBlock(self.deblur_flag,self.merge_flag, in_1=3*2,in_2=3*2, c=240)
        self.block1 = PMBBlock(self.deblur_flag,self.merge_flag, in_1=3*4+2*2+1,in_2=3*4, c=150)
        self.block2 = PMBBlock(self.deblur_flag,self.merge_flag, in_1=3*4+2*2+1,in_2=3*4, c=90)
        self.refine_flowT = Refine(in_num=16,out_num=5)
        self.temporal_boost = Boost(in_num=25,out_num=3)
    
    def forward(self, imgs_tensor, t_value,training=False, scale=[4,2,1]):
        '''
        imgs_tensor : (B,3*2,H,W)  [B0, B1] 
        t_value: (B,1)
        '''
        
        flow_list = []  ## different scale flows
        corrected_img_list = [] ## different scale corrected imgs
        occ_logit_list = []
        
        flow = None # flow:(batch_size,2*2,h,w)
        pmb_blocks = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow != None:
                
                corrected_img, flow, mask = pmb_blocks[i](imgs_tensor, corrected_img, flow,mask, scale=scale[i])  
            else:
                corrected_img, flow, mask = pmb_blocks[i](imgs_tensor, None, None,None, scale=scale[i]) # first block
            occ_logit_list.append(mask)
            flow_list.append(flow) 
            corrected_img_list.append(corrected_img)
        
        
        ##########  Predict flow_t0, flow_t1  #######
        
        # flow: [F_(S0->S1), F_(S1->S0)] (B,2*2,H,W);  corrected_img[-1]:[S0,S1]  (B,3*2,H,W)
        # mask : (B,1,H,W) 
        t_value = torch.unsqueeze(torch.unsqueeze(t_value, -1), -1)  # [B, 1, 1, 1]
        flow_t0, flow_t1 = CFR_flow_t_align(flow[:,:2], flow[:,2:], t_value) # CFR: Complementary Flow Reversal

        tt = t_value.repeat(1,1,flow.shape[-2],flow.shape[-1])
        tmp_res = self.refine_flowT(torch.cat([corrected_img[-1],flow,flow_t0,flow_t1,mask,tt],dim=1))
        occ_0_logit = mask + (tmp_res[:,4:5] * 2 - 1)
        flow_t1 = flow_t1 + (tmp_res[:,2:4] * 2 - 1)
        flow_t0 = flow_t0 + (tmp_res[:,:2] * 2 - 1)
        

        ####  Interpolate St with temporal boosting   ############
        
        occ_0 = torch.sigmoid(occ_0_logit)
        occ_1 = 1 - occ_0
        ## S0,F_t0  ->  St     warp(corrected_img[-1][:,:3],flow_t0 )  
        ## S1,F_t1  ->  St     warp(corrected_img[-1][:,3:], flow_t1)
        S0t = warp(corrected_img[-1][:,:3],flow_t0 )
        S1t = warp(corrected_img[-1][:,3:], flow_t1)  
        St = (1 - t_value) * occ_0 * S0t + t_value * occ_1 *  S1t
        St = St / ((1 - t_value) * occ_0 + t_value * occ_1) 
  
        B,C,H,W = St.shape
        t_value = t_value.repeat(1, 1, H, W)
        agg = torch.cat([corrected_img[-1],S0t,S1t,St,flow, t_value, flow_t0,flow_t1,occ_0_logit], dim=1)
        St_fined = St + (self.temporal_boost(agg)*2-1)
        return flow_list, corrected_img_list, St_fined, flow_t1, flow_t0
        

  
class PMBBlock(nn.Module):

    def __init__(self,deblur_flag,merge_flag, in_1,in_2,c=64):
        super(PMBBlock, self).__init__()
        
        num_resblock = 8
        if deblur_flag == 'mimo+':
            num_resblock = 16
        self.ifblock = IFBlock(in_1, c) 
        self.deblur = MIMOUNet(in_2, deblur_flag,num_res=num_resblock) 
        self.refine = Refine(in_num=16,out_num=3*2)
        if merge_flag=="normal":
            c = 16
            self.merge = nn.Sequential(
                      BasicConv(3*4, c, kernel_size=3, stride=1),
                      BasicConv(c, 2*c, kernel_size=3, stride=1),
                      BasicConv(2*c, c, kernel_size=3, stride=1),
                      BasicConv(c, 1, kernel_size=3, relu=False, stride=1)
                     )
        
        if merge_flag == "deformable":
            self.merge = FusionModule(3*2)

    def forward(self,x, corrected, flow, mask, scale):
        '''
        x:(B,6,h,w) [B0,B1]  corrected[-1]: (B,6,h,w) [S0,S1]   
        flow: (B,4,h,w)  [F_(S^(i)_(0)->S^(i-1)_(1)), F_(S^(i)_(1)->S^(i-1)_(0))]
        mask: (B,1,h,w)
        '''
        x = F.interpolate(x, scale_factor = 1. / scale, mode="bilinear", align_corners=False)
        
        if flow != None:
            flow = F.interpolate(flow, scale_factor = 2, mode="bilinear", align_corners=False) * 2
            mask = F.interpolate(mask, scale_factor = 2, mode="bilinear", align_corners=False)
            corrected =  F.interpolate(corrected[-1], scale_factor = 2, mode="bilinear", align_corners=False)
            x = torch.cat((x,corrected, flow,mask), 1)
        
        flow_d, mask_d = self.ifblock(x)  ### (B,4,h,w)
        
        
        ### GS blur-stream correction 
        if flow != None:
            flows = flow_d + flow
            masks = mask_d + mask
            corrected_list = self.deblur(x[:,:3*4], flows)
        else:
            flows = flow_d 
            masks = mask_d 
            corrected_list = self.deblur(x, flows)
            
        
        ##  RS effect-stream correction 
        # flows[:,:2] : F_(S^i_0->S^(i-1)_1)  flows[:,2:] : F_(S^i_1->S^(i-1)_0)
        warp_list = []
        if flow !=None:        
            # corrected[:,:3] : S^(i-1)_0               corrected[:,3:] : S^(i-1)_1
            warp_list.append(warp(corrected[:,3:], flows[:,:2]))  ## S^i_0
            warp_list.append(warp(corrected[:,:3], flows[:,2:]))  ## S^i_1
            warped_corrected_img = torch.cat(warp_list, dim=1)
            tmp = self.refine(torch.cat((corrected, warped_corrected_img, flows), 1))
        else:
            # x[:,:3] : S^(i-1)_0  (B_0)              x[:,3:] : S^(i-1)_1  (B_1)
            warp_list.append(warp(x[:,3:], flows[:,:2]))  ## S^i_0
            warp_list.append(warp(x[:,:3], flows[:,2:]))  ## S^i_1
            warped_corrected_img = torch.cat(warp_list, dim=1)
            tmp = self.refine(torch.cat((x, warped_corrected_img, flows), 1))  
        warped_corrected_img = warped_corrected_img + (tmp* 2 - 1)

        #Merge of corrected imgs from two paradigms 
        w0_logit = self.merge(torch.cat([corrected_list[-1],warped_corrected_img], dim=1))
        w0 = torch.sigmoid(w0_logit)
        merge_res = w0*corrected_list[-1] + (1-w0)*warped_corrected_img
        
        corrected_list[-1] = merge_res
        
        return  corrected_list, flows, masks
            
            
        
########################################################
'''
flow estimation part
'''
class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            BasicConv(in_planes, c//2, 3, 2),
            BasicConv(c//2, c, 3, 2),
            )
        self.convblock = nn.Sequential(
            BasicConv(c, c,3,1),
            BasicConv(c, c,3,1),
            BasicConv(c, c,3,1),
            BasicConv(c, c,3,1),
            BasicConv(c, c,3,1),
            BasicConv(c, c,3,1),
            BasicConv(c, c,3,1),
            BasicConv(c, c,3,1),
        )
        self.lastconv = BasicConv(c,4+1,4,2,relu=False,transpose=True)

    def forward(self, x):
    
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor = 2, mode="bilinear", align_corners=False)
        # flow: (batch_size,2*2,h,w)
        flow = tmp[:,:4] * 2
        mask = tmp[:, 4:5]
        return flow, mask



#####################################################
'''
 flow-guided feature alignment
''' 
class FFA(nn.Module):  
    def __init__(self, in_channel, out_channel):
        super(FFA, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4, flow):
        x = torch.cat([x1, x2, x4], dim=1)
        res = self.conv(x)  ## (B, C,H,W)
        warp_list = []
        divide = int(res.shape[1]/2)
        
        # res[:,:divide] : S^(i-1)_0               res[:,divide:] : S^(i-1)_1
        # flow[:,:2] : F_(S^i_0->S^(i-1)_1)  flow[:,2:] : F_(S^i_1->S^(i-1)_0)
        warp_list.append(warp(res[:,divide:], flow[:,:2]))  ## S^i_0
        warp_list.append(warp(res[:,:divide], flow[:,2:]))  ## S^i_1
        
        return torch.cat(warp_list, dim=1)


#################################################

'''
Unet-based refine network
'''
class Refine(nn.Module):
    def __init__(self, in_num,out_num, c=16):
        super(Refine, self).__init__()
        self.down0 = Conv2(in_num, 2*c)
        self.down1 = Conv2(2*c, 4*c)
        self.down2 = Conv2(4*c, 8*c)
        self.up0 = BasicConv(8*c, 4*c, kernel_size=4, stride=2, transpose=True)
        self.up1 = BasicConv(8*c, 2*c, kernel_size=4, stride=2, transpose=True)
        self.up2 = BasicConv(4*c, c, kernel_size=4, stride=2, transpose=True)
        self.conv = BasicConv(c, out_num, kernel_size=3, relu=False, stride=1)

    def forward(self, input_tensor):
        s0 = self.down0(input_tensor) #2c
        s1 = self.down1(s0) #4c
        s2 = self.down2(s1) #8c
        x = self.up0(s2) # 4c
        x = self.up1(torch.cat((x, s1), 1)) #2c
        x = self.up2(torch.cat((x, s0), 1)) #c
        x = self.conv(x) 
        return torch.sigmoid(x)


#######################################################       
'''
    debluring part: MIMOUNet
'''      

class MIMOUNet(nn.Module):
    def __init__(self, in_channel, deblur_flag, num_res=8):
        super(MIMOUNet, self).__init__()
        self.deblur_flag = deblur_flag
        self.in_channel = in_channel
        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_channel, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3*2, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])
        
        

        self.Convs = nn.ModuleList([
                BasicConv(base_channel * 2, base_channel * 2, kernel_size=1, relu=True, stride=1),  # base_channel * 4
                BasicConv(base_channel * 1, base_channel, kernel_size=1, relu=True, stride=1),      # base_channel * 2
            ])
        
            

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3*2, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3*2, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.FFAs = nn.ModuleList([
            FFA(base_channel * 7, base_channel*1),
            FFA(base_channel * 7, base_channel*2)
        ])
        
        
        if self.deblur_flag in ['mimo','mimo+']:
            self.FAM1 = FAM(base_channel * 4)
            self.SCM1 = SCM(self.in_channel, base_channel * 4)
            self.FAM2 = FAM(base_channel * 2)
            self.SCM2 = SCM(self.in_channel, base_channel * 2)

    def forward(self, x, flow):
        ###   fist block -- x [B0,B1]   otherwise, x [B0,B1,S0,S1]   (B,C,H,W)
    
        if self.deblur_flag in ['mimo','mimo+']:
            x_2 = F.interpolate(x, scale_factor=0.5)
            x_4 = F.interpolate(x_2, scale_factor=0.5)
            z2 = self.SCM2(x_2)
            z4 = self.SCM1(x_4)

        
        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)      ### base_channel*1   (B,C,H,W)
        

        z = self.feat_extract[1](res1)  ## downsampe 1/2    
        if self.deblur_flag in ['mimo','mimo+']:
            z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)       ### base_channel*2   (B,C,H/2,W/2)

        z = self.feat_extract[2](res2)  ## downsampe 1/4
        if self.deblur_flag in ['mimo','mimo+']:
            z = self.FAM1(z, z4)
        z = self.Encoder[2](z)          ## base_channel*4   (B,C,H/4,W/4) 
        

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)
        res1 = self.FFAs[0](res1, z21, z41,flow.detach())  ## base_channel*1   (B,C,H,W)
        flow = F.interpolate(flow, scale_factor = 1. / 2, mode="bilinear", align_corners=False) * 1. / 2
        res2 = self.FFAs[1](z12, res2, z42,flow.detach())  ## base_channel*2   (B,C,H/2,W/2)
        



        z = self.Decoder[0](z)
        if self.deblur_flag in ['mimo','mimo+']:
            z_ = self.ConvsOut[0](z)
            outputs.append(z_+x_4[:,-6:])
        z = self.feat_extract[3](z)  ## upsample 2   (B,C,H/2,W/2)
        

        
        
        #z = torch.cat([z, res2], dim=1)
        #z = self.Convs[0](z)
        z = z + self.Convs[0](z*res2)
        z = self.Decoder[1](z)
        if self.deblur_flag in ['mimo','mimo+']:
            z_ = self.ConvsOut[1](z)
            outputs.append(z_+x_2[:,-6:])
        z = self.feat_extract[4](z)  ## upsample 4  (B,C,H,W)
        

        
        #z = torch.cat([z, res1], dim=1)
        #z = self.Convs[1](z)
        z = z + self.Convs[1](z*res1)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x[:,-6:])

        return outputs


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
        

class SCM(nn.Module):
    def __init__(self, in_channel, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-in_channel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out



########################################################
'''
 temporal boosting module
'''
class Boost(nn.Module):
    def __init__(self, in_num, out_num, c=48):
        super(Boost, self).__init__()
        
        self.encode = nn.Sequential(
            BasicConv(in_num, c, kernel_size=5, stride=1),
            Conv2(c, 2*c),
        )
        G0 = 2*c
        G = 32
        C = 4
        self.num_RDB = 3 #6
        self.rdbs = nn.ModuleList()
        for i in range(self.num_RDB):
            self.rdbs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        self.decode= nn.Sequential(
             BasicConv(G0, c, kernel_size=4, stride=2, transpose=True),
             BasicConv(c, out_num, kernel_size=3, relu=False, stride=1),
        )
        
    def forward(self, x):
        z = self.encode(x)
        for i in range(self.num_RDB):
            z = self.rdbs[i](z)
        res = self.decode(z)
        return torch.sigmoid(res)



## Residual Dense Block
class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)       
        
        
#######################################################
'''
# CFR: Complementary Flow Reversal
'''
def CFR_flow_t_align(flow_01, flow_10, t_value):
    """ modified from https://github.com/JihyongOh/XVFI/blob/main/XVFInet.py"""
    ## Feature warping
    flow_01, norm0 = fwarp(flow_01,t_value * flow_01)  ## Actually, F (t) -> (t+1). Translation. Not normalized yet
    flow_10, norm1 = fwarp(flow_10, (1 - t_value) * flow_10)  ## Actually, F (1-t) -> (-t). Translation. Not normalized yet

    flow_t0 = -(1 - t_value) * (t_value) * flow_01 + (t_value) * (t_value) * flow_10
    flow_t1 = (1 - t_value) * (1 - t_value) * flow_01 - (t_value) * (1 - t_value) * flow_10

    norm = (1 - t_value) * norm0 + t_value * norm1
    mask_ = (norm.detach() > 0).type(norm.type())
    flow_t0 = (1 - mask_) * flow_t0 + mask_ * (flow_t0.clone() / (norm.clone() + (1 - mask_)))
    flow_t1 = (1 - mask_) * flow_t1 + mask_ * (flow_t1.clone() / (norm.clone() + (1 - mask_)))

    return flow_t0, flow_t1

def fwarp(img, flo):
    """
        -img: image (N, C, H, W)
        -flo: optical flow (N, 2, H, W)
        elements of flo is in [0, H] and [0, W] for dx, dy

    """

    # (x1, y1)		(x1, y2)
    # +---------------+
    # |				  |
    # |	o(x, y) 	  |
    # |				  |
    # |				  |
    # |				  |
    # |				  |
    # +---------------+
    # (x2, y1)		(x2, y2)

    N, C, _, _ = img.size()

    # translate start-point optical flow to end-point optical flow
    y = flo[:, 0:1:, :]
    x = flo[:, 1:2, :, :]

    x = x.repeat(1, C, 1, 1)
    y = y.repeat(1, C, 1, 1)

    # Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
    x1 = torch.floor(x)
    x2 = x1 + 1
    y1 = torch.floor(y)
    y2 = y1 + 1

    # firstly, get gaussian weights
    w11, w12, w21, w22 = get_gaussian_weights(x, y, x1, x2, y1, y2)

    # secondly, sample each weighted corner
    img11, o11 = sample_one(img, x1, y1, w11)
    img12, o12 = sample_one(img, x1, y2, w12)
    img21, o21 = sample_one(img, x2, y1, w21)
    img22, o22 = sample_one(img, x2, y2, w22)

    imgw = img11 + img12 + img21 + img22
    o = o11 + o12 + o21 + o22

    return imgw, o
    

def get_gaussian_weights(x, y, x1, x2, y1, y2):
    w11 = torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
    w12 = torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
    w21 = torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
    w22 = torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

    return w11, w12, w21, w22


def sample_one(img, shiftx, shifty, weight):
    
    """
    Input:
        -img (N, C, H, W)
        -shiftx, shifty (N, c, H, W)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    N, C, H, W = img.size()

    # flatten all (all restored as Tensors)
    flat_shiftx = shiftx.view(-1)
    flat_shifty = shifty.view(-1)
    flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(device).long().repeat(N, C,
                                                                                                          1,
                                                                                                          W).view(
        -1)
    flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(device).long().repeat(N, C,
                                                                                                          H,
                                                                                                          1).view(
        -1)
    flat_weight = weight.view(-1)
    flat_img = img.contiguous().view(-1)

    # The corresponding positions in I1
    idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(device).long().repeat(1, C, H, W).view(
        -1)
    idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(device).long().repeat(N, 1, H, W).view(
        -1)
    # ttype = flat_basex.type()
    idxx = flat_shiftx.long() + flat_basex
    idxy = flat_shifty.long() + flat_basey

    # recording the inside part the shifted
    mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

    # Mask off points out of boundaries
    ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
    ids_mask = torch.masked_select(ids, mask).clone().to(device)

    # Note here! accmulate fla must be true for proper bp
    img_warp = torch.zeros([N * C * H * W, ]).to(device)
    img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

    one_warp = torch.zeros([N * C * H * W, ]).to(device)
    one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

    return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)