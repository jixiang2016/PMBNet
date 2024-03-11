import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from math import exp
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EPE(nn.Module):
    def __init__(self):
        super(EPE, self).__init__()

    def forward(self, flow, gt, loss_mask):
        loss_map = (flow - gt.detach()) ** 2
        loss_map = (loss_map.sum(1, True) + 1e-6) ** 0.5
        return (loss_map * loss_mask)

class Ternary(nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape(
            (patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, img):
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm

    def rgb2gray(self, rgb):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def hamming(self, t1, t2):
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), 1, True)
        return dist_norm

    def valid_mask(self, t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, img0, img1):
        img0 = self.transform(self.rgb2gray(img0))
        img1 = self.transform(self.rgb2gray(img1))
        return self.hamming(img0, img1) * self.valid_mask(img0, 1)

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


# total variation loss for smoothing 
class SOBEL(nn.Module):
    def __init__(self):
        super(SOBEL, self).__init__()
        self.kernelX = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]).float()
        self.kernelY = self.kernelX.clone().T
        self.kernelX = self.kernelX.unsqueeze(0).unsqueeze(0).to(device)
        self.kernelY = self.kernelY.unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred, gt):
        N, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        img_stack = torch.cat(
            [pred.reshape(N*C, 1, H, W), gt.reshape(N*C, 1, H, W)], 0)
        sobel_stack_x = F.conv2d(img_stack, self.kernelX, padding=1)
        sobel_stack_y = F.conv2d(img_stack, self.kernelY, padding=1)
        pred_X, gt_X = sobel_stack_x[:N*C], sobel_stack_x[N*C:]
        pred_Y, gt_Y = sobel_stack_y[:N*C], sobel_stack_y[N*C:]

        L1X, L1Y = torch.abs(pred_X-gt_X), torch.abs(pred_Y-gt_Y)
        loss = (L1X+L1Y)
        return loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, rank=0):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        pretrained = True
        self.vgg_pretrained_features = models.vgg19(pretrained=pretrained).features
        self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X, Y, indices=None):
        X = self.normalize(X)
        Y = self.normalize(Y)
        indices = [2, 7, 12, 21, 30]
        weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        k = 0
        loss = 0
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            Y = self.vgg_pretrained_features[i](Y)
            if (i+1) in indices:
                loss += weights[k] * (X - Y.detach()).abs().mean() * 0.1
                k += 1
        return loss

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        #return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))
        return sum(charbonnier_loss(a, b) for a, b in zip(pyr_input, pyr_target))

def gauss_kernel(size=5, channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        #!!!
        if up.shape[-2:] != current.shape[-2:]:
            up = F.interpolate(up, size=current.shape[-2:], mode="bilinear", align_corners=False)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class Charbonnier_loss_color(nn.Module):
    """
    L1 Charbonnierloss color
    """

    def __init__(self, para):
        super(Charbonnier_loss_color, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        # print(diff_sq.shape)
        diff_sq_color = torch.mean(diff_sq, 1, True)
        # print(diff_sq_color.shape)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss

def charbonnier_loss(img1,img2):
    loss = CharbonnierLoss().to(device)
    return loss(img1,img2)


#### Differentiable implementation version of structural similarity (SSIM) index for SSIM loss
#### SSIMLoss =  pytorch_ssim.SSIM()   loss_ssim = 1 - SSIMLoss(img1, img2)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)




##### adversarial loss 
class GAN(nn.Module):
    """
    GAN loss
    """

    def __init__(self, para):
        super(GAN, self).__init__()
        self.device = torch.device('cpu' if para.cpu else 'cuda')
        self.D = Discriminator().to(self.device)
        self.criterion = nn.BCELoss().to(self.device)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=para.lr)
        self.real_label = 1
        self.fake_label = 0

    def forward(self, x, y, valid_flag=False):
        self.D.zero_grad()
        b = y.size(0)
        label = torch.full((b,), self.real_label, device=self.device)
        if not valid_flag:
            ############################################
            # update D network: maximize log(D(y) + log(1-D(G(x))))
            # train with all-real batch
            output = self.D(y).view(-1)  # forward pass of real batch through D
            errD_real = self.criterion(output, label)  # calculate loss on all-real batch
            errD_real.backward()  # calculate gradients for D in backward pass
            ## D_y = output.mean().item()
            # train with all-fake batch
            label.fill_(self.fake_label)
            output = self.D(x.detach()).view(-1)  # classify all fake batch with D
            errD_fake = self.criterion(output, label)  # calculate D's loss on the all-fake batch
            errD_fake.backward()  # calculate gradients for all-fake batch
            ## D_G_x1 = output.mean().item()
            # add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # update D
            self.D_optimizer.step()
        ############################################
        # generate loss for G network: maximize log(D(G(x)))
        label.fill_(self.real_label)  # fake labels are all real for generator cost
        # since we just updated D, perform another forward pass of all-fake batch through D
        output = self.D(x).view(-1)
        errG = self.criterion(output, label)  # calculate G's loss on this output
        ## D_G_x2 = output.mean().item()

        return errG

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # x: (n,3,256,256)
        c = 3
        h, w = 256, 256
        n_feats = 8
        n_middle_blocks = 6
        self.start_module = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=n_feats, kernel_size=3, stride=1, padding=1),  # (n,8,256,256)
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            BasicBlock(in_channels=n_feats, out_channels=n_feats, kernel_size=3, stride=2, padding=1),  # (n,8,128,128)
        )
        middle_module_list = []
        for i in range(n_middle_blocks):
            middle_module_list.append(
                BasicBlock(in_channels=n_feats * (2 ** i), out_channels=n_feats * (2 ** (i + 1)), kernel_size=3,
                           stride=1, padding=1))
            middle_module_list.append(
                BasicBlock(in_channels=n_feats * (2 ** (i + 1)), out_channels=n_feats * (2 ** (i + 1)), kernel_size=3,
                           stride=2, padding=1))
        self.middle_module = nn.Sequential(*middle_module_list)
        self.end_module = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (n,3,256,256)
        n, _, _, _ = x.shape
        h = self.start_module(x)  # (n,8,256,256)
        h = self.middle_module(h)  # (n,512,2,2)
        h = h.reshape(n, -1)  # (n,2048)
        out = self.end_module(h)  # (n,1)
        return out



if __name__ == '__main__':
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(
        0, 1, (3, 3, 256, 256))).float().to(device)
    ternary_loss = Ternary()
    print(ternary_loss(img0, img1).shape)
