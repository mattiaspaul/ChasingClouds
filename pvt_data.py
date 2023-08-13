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

from tqdm import trange
#from divroc_utils import *

#from divroc import *
from ppwc import computeChamfer

import torch
import pyvista as pv
import os

import numpy as np
#import pyvista as pv

import nibabel as nib 





def read_vtk(path):
    data = pv.read(path)
    data_dict = {}
    data_dict["points"] = data.points.astype(np.float32)
    data_dict["faces"] = data.faces.reshape(-1, 4)[:, 1:].astype(np.int32)
    for name in data.array_names:
        try:
            data_dict[name] = data[name]
        except:
            pass
    return data_dict


#density estimation on knn-graph followed by non-max suppression to obtain ~8192 points per cloud
def foerstner_nms(pcd, sigma, neigh_1, neigh_2, min_points, sigma_interval):
    pcd = pcd.cuda()
    knn = torch.zeros(len(pcd), neigh_1).long().cuda()
    knn_dist = torch.zeros(len(pcd), neigh_1).float().cuda()
    with torch.no_grad():
        chk = torch.chunk(torch.arange(len(pcd)).cuda(), 192)
        for i in trange(len(chk)):
            dist = (pcd[chk[i]].unsqueeze(1) - pcd.unsqueeze(0)).pow(2).sum(-1).sqrt()
            q = torch.topk(dist, neigh_1, dim=1, largest=False)
            knn[chk[i]] = q[1][:, :]
            knn_dist[chk[i]] = q[0][:, :]

    curr_points = 0
    curr_sigma = sigma
    while curr_points < min_points:
        exp_score = torch.exp(-knn_dist[:, :].pow(2) * curr_sigma ** 2).mean(1)
        knn_score = torch.max(exp_score[knn[:, :neigh_2]], 1)[0]
        valid_idx = (knn_score == exp_score).nonzero(as_tuple=True)[0]
        curr_points = valid_idx.shape[0]
        curr_sigma += sigma_interval
    return valid_idx,knn,knn_dist#rand_idx[valid_idx]


def load_training_data(pix,dims):
    files = sorted(os.listdir('pvt1010/PVT1010/'))
    files = files[20:] #leave-out DIRLAB-COPD validation cases!
    count = 0

    files_new = []
    for i in range(len(files)):
        elem = files[i]
        if('EXP' in elem):
            elem = elem.replace('EXP','INSP')
        else:
            elem = elem.replace('INSP','EXP')
        idx = files.index(elem) if elem in files else -1
        if(idx>=0):
            count += 1
            if('EXP' in files[i]):
                files_new.append(files[i])

    pvt_1k_cases = []
    pvt_1k_exp = []
    pvt_1k_insp = []


    for i in trange(len(files_new)):
        case = int(files_new[i].split('_')[1])
        pcd_exp = read_vtk('pvt1010/PVT1010/copd_{:06d}_EXP.vtk'.format(case))
        pcd_exp = torch.tensor(pcd_exp['points'])

        pcd_insp = read_vtk('pvt1010/PVT1010/copd_{:06d}_INSP.vtk'.format(case))
        pcd_insp = torch.tensor(pcd_insp['points'])


        pix_insp_exp = pix.mean(0)[:,0]
        dim_insp_exp = dims.mean(0)[:,0].long()



        x1 = (pcd_exp[:,0]/pix_insp_exp[0]).float()/dim_insp_exp[0]*2-1
        y1 = (pcd_exp[:,1]/pix_insp_exp[1]).float()/dim_insp_exp[1]*2-1
        z1 = (pcd_exp[:,2]/pix_insp_exp[2]).float()/dim_insp_exp[2]*2-1
        keypts1_exp = torch.stack((z1,y1,x1),1)
        keypts1_exp -= keypts1_exp.mean(0,keepdim=True)
        x1 = (pcd_insp[:,0]/pix_insp_exp[0]).float()/dim_insp_exp[0]*2-1
        y1 = (pcd_insp[:,1]/pix_insp_exp[1]).float()/dim_insp_exp[1]*2-1
        z1 = (pcd_insp[:,2]/pix_insp_exp[2]).float()/dim_insp_exp[2]*2-1
        keypts1_insp = torch.stack((z1,y1,x1),1)
        keypts1_insp -= keypts1_insp.mean(0,keepdim=True)

        pvt_1k_cases.append(case)
        pvt_1k_exp.append(keypts1_exp)
        pvt_1k_insp.append(keypts1_insp)
    return pvt_1k_exp,pvt_1k_insp,pvt_1k_cases

def load_validation_data():

    #constants
    COPD_info = {"copd1": {"insp":{'size': [512, 512, 482],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -310.625]},
                        "exp":{'size': [512, 512, 473],'spacing': [0.625, 0.625, 0.625], 'origin': [-148.0, -145.0, -305.0]}},
              "copd2":  {"insp":{'size': [512, 512, 406],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-176.9, -165.0, -254.625]},
                        "exp":{'size': [512, 512, 378],'spacing': [0.644531, 0.644531, 0.625], 'origin': [-177.0, -165.0, -237.125]}},
              "copd3":  {"insp":{'size': [512, 512, 502],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -343.125]},
                        "exp":{'size': [512, 512, 464],'spacing': [0.652344, 0.652344, 0.625], 'origin': [-149.4, -167.0, -319.375]}},
              "copd4":  {"insp":{'size': [512, 512, 501],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -308.25]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.589844, 0.589844, 0.625], 'origin': [-124.1, -151.0, -283.25]}},
              "copd5":  {"insp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]},
                        "exp":{'size': [512, 512, 522],'spacing': [0.646484, 0.646484, 0.625], 'origin': [-145.9, -175.9, -353.875]}},
              "copd6":  {"insp":{'size': [512, 512, 474],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -299.625]},
                        "exp":{'size': [512, 512, 461],'spacing': [0.632812, 0.632812, 0.625], 'origin': [-158.4, -162.0, -291.5]}},
              "copd7":  {"insp":{'size': [512, 512, 446],'spacing': [0.625, 0.625, 0.625], 'origin': [-150.7, -160.0, -301.375]},
                        "exp":{'size': [512, 512, 407],'spacing': [0.625, 0.625, 0.625], 'origin': [-151.0, -160.0, -284.25]}},
              "copd8":  {"insp":{'size': [512, 512, 458],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -313.625]},
                        "exp":{'size': [512, 512, 426],'spacing': [0.585938, 0.585938, 0.625], 'origin': [-142.3, -147.4, -294.625]}},
              "copd9":  {"insp":{'size': [512, 512, 461],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -310.25]},
                        "exp":{'size': [512, 512, 380],'spacing': [0.664062, 0.664062, 0.625], 'origin': [-156.1, -170.0, -259.625]}},
              "copd10": {"insp":{'size': [512, 512, 535],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -355.0]},
                        "exp":{'size': [512, 512, 539],'spacing': [0.742188, 0.742188, 0.625], 'origin': [-189.0, -176.0, -346.25]}}
              }
    orig_img_z = torch.tensor([484., 408., 504., 504., 524., 476., 448., 460., 464., 540])

    orig_img_dim = torch.cat((torch.ones(10,2)*512,orig_img_z.view(10,1)),1).unsqueeze(-1).repeat(1,1,2)

    pvt_img_dim = torch.zeros(10,3,2)
    pvt_img_origin = torch.zeros(10,3,2)
    for case in trange(1,11):
        info_exp = COPD_info['copd'+str(case)]['exp']
        info_insp = COPD_info['copd'+str(case)]['insp']

        dim_exp = torch.tensor(info_exp['size'])
        dim_insp = torch.tensor(info_insp['size'])
        pvt_img_dim[case-1,:,1] = dim_insp
        pvt_img_dim[case-1,:,0] = dim_exp


        origin_exp = torch.tensor(info_exp['origin'])
        origin_insp = torch.tensor(info_insp['origin'])
        pvt_img_origin[case-1,:,1] = origin_insp
        pvt_img_origin[case-1,:,0] = origin_exp
    dims = torch.zeros(10,3,2)
    origin = torch.zeros(10,3,2)
    pix = torch.zeros(10,3,2)
    centre = torch.zeros(10,3,2)
    
    
    pvt_dense_exp = []
    pvt_dense_insp = []
    pvt_dense_orig_exp = []
    pvt_dense_orig_insp = []
    indices_all = []
    for case in trange(1,11):

        indices = torch.load('pvtcopd_vtk/case00'+str(case).zfill(2)+'_indices.pth')
        indices_all.append(indices)
        pcd_exp = read_vtk('pvtcopd_vtk/copd_{:06d}_EXP.vtk'.format(case))
        pcd_exp = torch.tensor(pcd_exp['points'])

        pcd_insp = read_vtk('pvtcopd_vtk/copd_{:06d}_INSP.vtk'.format(case))
        pcd_insp = torch.tensor(pcd_insp['points'])

        info_exp = COPD_info['copd'+str(case)]['exp']
        pix_exp = info_exp['spacing']
        dim_exp = info_exp['size']
        origin_exp = info_exp['origin']


        info_insp = COPD_info['copd'+str(case)]['insp']
        pix_insp = info_insp['spacing']
        #done fixed problem with different cropping dimensions in insp/exp 
        dim_insp = info_insp['size']#info_insp['size']
        origin_insp = info_insp['origin']#info_insp['origin']
        pcd_exp_orig = pcd_exp.clone()
        pcd_insp_orig = pcd_insp.clone()

        pvt_dense_orig_exp.append(pcd_exp_orig)
        pvt_dense_orig_insp.append(pcd_insp_orig)


        pix[case-1,:,0] = torch.tensor(pix_exp)
        pix[case-1,:,1] = torch.tensor(pix_insp)

        origin[case-1,:,0] = torch.tensor(origin_exp)
        origin[case-1,:,1] = torch.tensor(origin_insp)

        dims[case-1,:,0] = torch.tensor(dim_exp)
        dims[case-1,:,1] = torch.tensor(dim_insp)


        orig_dim = orig_img_dim[case-1,:,0]
        pcd_exp[:,2] += (orig_img_dim[case-1,2,0]-int(dim_exp[2]))*pix_exp[2]


        pcd_insp[:,2] += (orig_img_dim[case-1,2,0]-int(dim_insp[2]))*pix_insp[2]

        centre[case-1,:,0] = pcd_exp.mean(0)
        centre[case-1,:,1] = pcd_insp.mean(0)
        idx_exp = indices['tgt_exp_idx']
        idx_insp = indices['src_insp_idx']


        pvt_sparse_insp = pcd_insp_orig[idx_insp]
        
        x1 = (pcd_exp[:,0]/pix_exp[0]-origin_exp[0]/pix_exp[0]).float().cuda()/dim_exp[0]*2-1
        y1 = (pcd_exp[:,1]/pix_exp[1]-origin_exp[1]/pix_exp[1]).float().cuda()/dim_exp[1]*2-1
        z1 = (pcd_exp[:,2]/pix_exp[2]-origin_exp[2]/pix_exp[2]).float().cuda()/dim_exp[2]*2
        ratio = float(dim_exp[2]/2)/float(orig_dim[2]/2)
        keypts1_exp = torch.stack((z1*ratio-1,y1,x1),1)


        x1 = (pcd_insp[:,0]/pix_insp[0]-origin_insp[0]/pix_insp[0]).float().cuda()/dim_insp[0]*2-1
        y1 = (pcd_insp[:,1]/pix_insp[1]-origin_insp[1]/pix_insp[1]).float().cuda()/dim_insp[1]*2-1
        z1 = (pcd_insp[:,2]/pix_insp[2]-origin_insp[2]/pix_insp[2]).float().cuda()/dim_insp[2]*2
        ratio = float(dim_insp[2]/2)/float(orig_dim[2]/2)
        keypts1_insp = torch.stack((z1*ratio-1,y1,x1),1)

        pvt_dense_exp.append(keypts1_exp)
        pvt_dense_insp.append(keypts1_insp)



    dim_insp = torch.zeros(10,3)
    dim_exp = torch.zeros(10,3)

    for case in range(1,11):
        info1 = COPD_info['copd'+str(case)]['exp']
        info_insp = COPD_info['copd'+str(case)]['insp']
        dim_insp[case-1,:] = torch.tensor(info_insp['size'])
        dim_exp[case-1,:] = torch.tensor(info1['size'])

    lm_exp0 = []
    pvt_copd_lm_insp = []
    pvt_copd_dim = []
    pvt_copd_spacing = []
    for case in range(1,11):
        lms_exp = np.loadtxt('COPDgene_landmarks/copd'+str(case)+'_300_eBH_xyz_r1.txt')
        lms_insp = np.loadtxt('COPDgene_landmarks/copd'+str(case)+'_300_iBH_xyz_r1.txt')
        dim1 = orig_img_dim[case-1,:,0]/torch.tensor([1,1,4])-1#torch.from_numpy(nib.load('COPDgene/COPD'+str(case).zfill(2)+'_img_fixed.nii.gz').header['dim'][1:4]).float()-1
        dim0 = orig_img_dim[case-1,:,0]/torch.tensor([1,1,4])-1#dim0 = torch.from_numpy(nib.load('COPDgene/COPD'+str(case).zfill(2)+'_img_fixed.nii.gz').header['dim'][1:4]).float()-1
        dim0 /= torch.tensor([2,2,.5])

        spacing = 2*torch.tensor(COPD_info['copd'+str(case)]['exp']['spacing'])
        shape = torch.tensor(COPD_info['copd'+str(case)]['exp']['size'])/2

        lms_insp1 = (lms_insp/torch.tensor([dim1[0]/2,dim1[1]/2,dim1[2]/2]).view(1,3)-1).flip(-1).cuda().float()*torch.tensor([-1,1,1]).cuda().view(1,3)
        lms_exp1 = (lms_exp/torch.tensor([dim1[0]/2,dim1[1]/2,dim1[2]/2]).view(1,3)-1).flip(-1).cuda().float()*torch.tensor([-1,1,1]).cuda().view(1,3)
        pvt_copd_lm_insp.append(lms_insp1)
        lm_exp0.append(lms_exp1)
        pvt_copd_dim.append(dim0)
        spacing = 2*torch.tensor(COPD_info['copd'+str(case)]['exp']['spacing'])

        pvt_copd_spacing.append(spacing)


    return pvt_dense_exp,pvt_dense_insp,indices_all,pvt_copd_lm_insp,lm_exp0,pvt_copd_dim,pvt_copd_spacing

    

    
