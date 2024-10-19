### instant_ngp
ref: https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/tree/main/Instant_Neural_Graphics_Primitives_with_a_Multiresolution_Hash_Encoding

ds: https://drive.google.com/drive/folders/1eO7DXFhWWpauC-9LDhOimtIKxY3yRCIm \
in the nerf paper they used spherical harmonics for color instead
#### TODO
- [ ] dataset class
- [ ] tcnn hashing
- [ ] tcnn forward
- [ ] bindingcuda
- [ ] cuda loss
- [ ] cuda ray marching
- [ ] implement a viewer for training

### setup 
```sh
conda install -c conda-forge cuda=12.1 gxx python=3.11.8
pip3 install torch torchvision torchaudio #torch 2.4.1 cu12.1
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install einops kornia matplotlib opencv-python lpips imageio imageio-ffmpeg scipy pymcubes trimesh dearpygui lightning
```
