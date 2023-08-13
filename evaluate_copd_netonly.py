#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torch.utils.checkpoint import checkpoint
import time
#import matplotlib.pyplot as plt

from torch.autograd import Function
from torch.autograd.functional import jacobian
device = 'cuda'

#torch.backends.cuda.matmul.allow_tf32 = False
#torch.backends.cudnn.allow_tf32 = False

from tqdm import trange, tqdm

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import os
import sys
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '4'
#!nvidia-smi


sys.path.insert(0,'point_pwc/')
from pointconv_util import *
from models import *
from pvt_data import *


# In[8]:


class TPS:
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device

        n = c.shape[0]
        f_dim = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, 4), device=device)
        P[:, 1:] = c

        v = torch.zeros((n + 4, f_dim), device=device)
        v[:n, :] = f

        A = torch.zeros((n + 4, n + 4), device=device)
        A[:n, :n] = K
        A[:n, -4:] = P
        A[-4:, :n] = P.t()

        theta = torch.linalg.solve(A, v)
        # theta = torch.solve(v, A)[0]
        return theta

    @staticmethod
    def d(a, b):
        ra = (a ** 2).sum(dim=1).view(-1, 1)
        rb = (b ** 2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return (r ** 2) * torch.log(r + 1e-6)

    @staticmethod
    def z(x, c, theta):
        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-4], theta[-4:].unsqueeze(2)
        b = torch.matmul(U, w)
        return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()


def thin_plate_dense(x1, y1, shape, step, lambd=.0, unroll_step_size=2 ** 12):
    device = x1.device
    D, H, W = shape
    D1, H1, W1 = D // step, H // step, W // step

    x2 = F.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), (1, 1, D1, H1, W1), align_corners=True).view(-1, 3)
    tps = TPS()
    theta = tps.fit(x1[0], y1[0], lambd)

    y2 = torch.zeros((1, D1 * H1 * W1, 3), device=device)
    N = D1 * H1 * W1
    n = math.ceil(N / unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y2[0, j1:j2, :] = tps.z(x2[j1:j2], x1[0], theta)

    y2 = y2.view(1, D1, H1, W1, 3).permute(0, 4, 1, 2, 3)
    y2 = F.interpolate(y2, (D, H, W), mode='trilinear', align_corners=True).permute(0, 2, 3, 4, 1)

    return y2


# In[9]:





# In[10]:


def main():
    
    pvt_dense_exp,pvt_dense_insp,indices_all,pvt_copd_lm_insp,lm_exp0,pvt_copd_dim,pvt_copd_spacing = load_validation_data()

    ppwc,run_loss_ = torch.load('models/divroc_ppwc_selftrain.pth'); lin_embed0 = None; model0 = None;


    scale_ppwc=47
    #EVALUATE ONLY NETWORK PREDICTION FIRST
    tre_net = torch.zeros(10,300)
    tre_before = torch.zeros(10,300)
    for ii in trange(10):
        indices = indices_all[ii]
        kpts_fix_sparse = pvt_dense_insp[ii].view(1,-1,1,1,3)[:,indices['src_insp_idx']].cuda()
        kpts_fix_dense = pvt_dense_insp[ii].view(1,-1,1,1,3).cuda()

        with torch.no_grad():
            indices = indices_all[ii]
            kpts_fixed = pvt_dense_insp[ii].view(1,-1,3).cuda()[:,indices['src_insp_idx']]
            kpts_moving = pvt_dense_exp[ii].view(1,-1,3).cuda()[:,indices['tgt_exp_idx']]


            kpts_fixed_dense = pvt_dense_insp[ii].view(1,-1,3).cuda()#[:,indices['src_insp_idx']]


            cloud_exp = pvt_dense_exp[ii].view(1,-1,3).cuda()
            N_fix = kpts_fixed.squeeze().shape[0]
            scale = 19
            scale = scale_ppwc
            out = torch.zeros(1,N_fix,3).cuda()
            for r in range(8):
                kpts_moving = cloud_exp[:,torch.randperm(cloud_exp.shape[1])[:8192]]
                out += .125*torch.tanh(ppwc(kpts_fixed, kpts_moving, torch.ones_like(kpts_fixed), torch.ones_like(kpts_moving))[0][0]).permute(0,2,1)

        grid_fix = kpts_fixed_dense.view(1,-1,1,1,3).cuda()

        with torch.no_grad():
            dense_flow_ = thin_plate_dense(kpts_fixed, out[0, :, :3].cuda().view(1,-1,3), (128,128,128), 4, 0.1)

        dense_flow = F.avg_pool3d(F.avg_pool3d(F.interpolate(dense_flow_.permute(0,4,1,2,3),size=(scale,scale,scale),mode='trilinear'),5,stride=1,padding=2),5,stride=1,padding=2)#.data


        kpts_mov_dense = pvt_dense_exp[ii].view(1,-1,1,1,3).cuda()
        kpts_mov_sparse = pvt_dense_exp[ii].view(1,-1,1,1,3).cuda()[:,indices['tgt_exp_idx']]

        dim0 = pvt_copd_dim[ii]
        lms_insp1 = pvt_copd_lm_insp[ii]
        lm_exp_align = lm_exp0[ii]
        spacing = pvt_copd_spacing[ii]

        flow_disp_net = F.grid_sample(dense_flow_.permute(0,4,1,2,3),lms_insp1.view(1,-1,1,1,3)).squeeze().t()
        tre0 = ((lm_exp_align-(lms_insp1))*torch.tensor([dim0[2]/2,dim0[1]/2,dim0[0]/2]).cuda()*spacing.cuda()).pow(2).sum(1).sqrt()
        tre_before[ii,:] = tre0#.mean()

        tre1 = ((lm_exp_align-(lms_insp1+flow_disp_net))*torch.tensor([dim0[2]/2,dim0[1]/2,dim0[0]/2]).cuda()*spacing.cuda()).pow(2).sum(1).sqrt()
        tre_net[ii,:] = tre1#.mean()

    print('TRE before '+'%0.3f'%tre_before.mean()+' mm -> network only '+'%0.3f'%tre_net.mean(),' mm')



# In[ ]:


if __name__ == '__main__':
	main()

