to run inference with pretrained PPWC models some pre-requisites need to be installed 

specifically you need a matching cuda-toolkit to your nvidia-driver and pytorch version
e.g. when you installed 
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
you also need to install
```
mamba install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

Next the pointlib2 needs to be compiled with ``point_pwc/pointnet2/lib; python setup.py install``
