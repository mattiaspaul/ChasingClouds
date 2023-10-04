## Chasing clouds: Differentiable volumetric rasterisation of point clouds as a highly efficient and accurate loss for large-scale deformable 3D registration
Code, models and dataset for ICCV 2023 (oral) paper on differentiable volumetric rasterisation of point clouds for 3D registration

Trained models can be found in the subdirectory models, inference, binaries for Adam optimisation and evaluation code for the geometric networks is already available. Open source code for instance optimisation and training of models will follow soon (at latest in time for ICCV in October 2023).

See open open job opportunities PostDoc and/or PhD at <http://mpheinrich.de/opportunities.html>

Check out an interactive live-demo or apply DiVRoC on your own data at <https://huggingface.co/spaces/mattiaspaul/chasingclouds>

# Datasets
The newly created LCSD500 dataset (paired vessel point clouds) is available for download at https://cloud.imi.uni-luebeck.de/s/mtPmXoeNrnntNBg (170MB), automatic correspondences are placed in LCSD500_keypoints_corrfield.zip.
More information on how to obtain the PVT_COPD dataset used for training and evaluation of our models can be found in pvtcopd_vtk/README.md

# Quick Start

```
git clone https://github.com/mattiaspaul/ChasingClouds.git
virutalenv venv
source venv/bin/activate
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip3 install -r requirements.txt
cd point_pwc/pointnet2/lib; python setup.py install; cd ../../..
mkdir predict
```
Follow instructions to download required files from public datasets <https://github.com/mattiaspaul/ChasingClouds/blob/main/pvtcopd_vtk/README.md>

Next you can run the PointPWC model using ``python evaluate_copd_netonly.py -O predict``. 
Note that the standalone DiVRoC regularisation (which improves TRE slightly) will be added soon.
Instance optimisation is available as binary by calling ``python dist/divroc_adam.py -O predict``, which improves the PPWC predictions to about **2.37 mm** TRE (state-of-the-art for PVT1010)! The source code will be released once IP clearance is ready (end September 2023).
### Features
- [x] Dataset (LCSD500 and instructions for PVT1010)
- [x] Environment and guide to compile PPWC library
- [x] Base model(s) (pth) and code for inference
- [x] TRE evaluation (without visualisation) 
- [x] Binaries for inference with Adam instance optimisation
- [x] Source code for DiVRoC
- [ ] Source code for distance/regularisation incl. training
- [ ] More models and DiVRoC ablations
- [ ] Visualisation and training code

# Paper
**by Mattias P. Heinrich, Alexander Bigalke, Christoph Großbröhmer (Uni Lübeck) and Lasse Hansen (EchoScout GmbH)**

Learning-based registration for large-scale 3D point clouds has been shown to improve robustness and accuracy compared to classical methods and can be trained without supervision for locally rigid problems. However, for tasks with highly deformable structures, such as alignment of pulmonary vascular trees for medical diagnostics, previous approaches of self-supervision with regularisation and point distance losses have failed to succeed, leading to the need for complex synthetic augmentation strategies to obtain reliably strong supervision. In this work, we introduce a novel Differentiable Volumetric Rasterisation of point Clouds (DiVRoC) that overcomes those limitations and offers a highly efficient and accurate loss for large-scale deformable 3D registration. DiVRoC drastically reduces the computational complexity for measuring point cloud distances for high-resolution data with over 100k 3D points and can also be employed to extrapolate and regularise sparse motion fields, as loss in a self-training setting and as objective function in instance optimisation. DiVRoC can be successfully embedded into geometric registration networks, including PointPWC-Net and other graph CNNs. Our approach yields new state-of-the-art accuracy on the challenging PVT dataset in three different settings without training with manual ground truth: 1) unsupervised metric-based learning 2) self-supervised learning with pseudo labels generated by self-training and 3) optimisation based alignment without learning.

![Concept](iccv_fig1.png?raw=true "Concept")

# Visual and numerical results on PVT1010 COPD dataset
![Results](github_visual.png?raw=true "Concept")

