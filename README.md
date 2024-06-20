### AIME Fork of Stable Diffusion 3 Micro-Reference Implementation

- Ready to use as a worker for the [AIME API Server](https://github.com/aime-team/aime-api-server)
- Added possibility to generate multiple images at once
- Preview images while processing
- Text to Image and Image to Image


### Download weigths

```shell
sudo apt-get install git-lfs
git lfs install
mkdir /destination/to/checkpoints
cd /destination/to/checkpoints
git clone https://huggingface.co/stabilityai/stable-diffusion-3-medium
```

### Clone this repo
```shell
cd /destination/to/repo
git clone https://github.com/aime-labs/stable_diffusion_3
```

### Setting up AIME MLC
```shell

mlc-create sd3 Pytorch 2.3.1-aime -d="/destination/to/checkpoints" -w="/destination/to/repo"
```
The -d flag will mount /destination/to/checkpoints to /data in the container. 

The -w flag will mount /destination/to/repo to /workspace in the container.


### Install requirements in AIME MLC
```shell
mlc-open sd3

pip install -r /workspace/stable_diffusion_3/requirements.txt

```

### Run SD3 inference as HTTP/HTTPS API with AIME API Server

To run Stable Diffusion XL as HTTP/HTTPS API with [AIME API Server](https://github.com/aime-team/aime-api-server) start the chat command with following command line:

```shell
mlc-open sd3
python3 /workspace/stable_diffusion_3/main.py --api_server <url to API server> --ckpt_dir /data/stable-diffusion-3-medium
```

It will start Stable Diffusion 3 as worker, waiting for job request through the AIME API Server.


### File Guide

- `sd3_infer.py` - entry point, review this for basic usage of diffusion model and the triple-tenc cat
- `sd3_impls.py` - contains the wrapper around the MMDiT and the VAE
- `other_impls.py` - contains the CLIP model, the T5 model, and some utilities
- `mmdit.py` - contains the core of the MMDiT itself
- folder `models` with the following files (download separately):
    - `clip_g.safetensors` (openclip bigG, same as SDXL, can grab a public copy)
    - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL, can grab a public copy)
    - `t5xxl.safetensors` (google T5-v1.1-XXL, can grab a public copy)
    - `sd3_medium.safetensors` (or whichever main MMDiT model file)



MIT License

Copyright (c) 2024 Stability AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
