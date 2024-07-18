# NOTE: Must have folder `models` with the following files:
# - `clip_g.safetensors` (openclip bigG, same as SDXL)
# - `clip_l.safetensors` (OpenAI CLIP-L, same as SDXL)
# - `t5xxl.safetensors` (google T5-v1.1-XXL)
# - `sd3_medium.safetensors` (or whichever main MMDiT model file)
# Also can have
# - `sd3_vae.safetensors` (holds the VAE separately if needed)

import torch, math
from safetensors import safe_open
from .other_impls import SDClipModel, SDXLClipG, T5XXLModel, SD3Tokenizer
from .sd3_impls import BaseModel, sample_euler, SDVAE, CFGDenoiser, SD3LatentFormat
from PIL import Image
import numpy as np


#################################################################################################
### Wrappers for model parts
#################################################################################################


def load_into(f, model, prefix, device, dtype=None):
    """Just a debugging-friendly hack to apply the weights in a safetensors file to the pytorch module."""
    for key in f.keys():
        if key.startswith(prefix) and not key.startswith("loss."):
            path = key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(f"Skipping key '{key}' in safetensors file as '{p}' does not exist in python model")
                        break
            if obj is None:
                continue
            try:
                tensor = f.get_tensor(key).to(device=device)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32
}


class ClipG:
    def __init__(self, model):
        with safe_open(model / "text_encoders/clip_g.safetensors", framework="pt", device="cpu") as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12
}


class ClipL:
    def __init__(self, model):
        with safe_open(model / "text_encoders/clip_l.safetensors", framework="pt", device="cpu") as f:
            self.model = SDClipModel(layer="hidden", layer_idx=-2, device="cpu", dtype=torch.float32, layer_norm_hidden_state=False, return_projected_pooled=False, textmodel_json_config=CLIPL_CONFIG)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128
}


class T5XXL:
    def __init__(self, model):
        with safe_open(model / "text_encoders/t5xxl_fp16.safetensors", framework="pt", device="cpu") as f:
            self.model = T5XXLModel(T5_CONFIG, device="cpu", dtype=torch.float32)
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


class SD3:
    def __init__(self, model, shift):
        with safe_open(model / 'sd3_medium.safetensors', framework="pt", device="cpu") as f:
            self.model = BaseModel(shift=shift, file=f, prefix="model.diffusion_model.", device="cpu", dtype=torch.float16).eval()
            load_into(f, self.model, "model.", "cpu", torch.float16)


class VAE:
    def __init__(self, model):
        with safe_open(model / 'sd3_medium.safetensors', framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=torch.float16).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", torch.float16)


#################################################################################################
### Main inference logic
#################################################################################################


# Note: Sigma shift value, publicly released models use 3.0
SHIFT = 3.0
# Naturally, adjust to the width/height of the model you have
WIDTH = 1024
HEIGHT = 1024
# Pick your prompt
PROMPT = "a photo of a cat"
# Most models prefer the range of 4-5, but still work well around 7
CFG_SCALE = 5
# Different models want different step counts but most will be good at 50, albeit that's slow to run
# sd3_medium is quite decent at 28 steps
STEPS = 50
# Random seed
SEED = 1
# Actual model file path
MODEL = "models/sd3_medium.safetensors"
# VAE model file path, or set None to use the same model file
VAEFile = None # "models/sd3_vae.safetensors"
# Optional init image file path
INIT_IMAGE = None
# If init_image is given, this is the percentage of denoising steps to run (1.0 = full denoise, 0.0 = no denoise at all)
DENOISE = 0.6
# Output file path
OUTPUT = "output.png"

class SD3Inferencer:
    def load(self, model=MODEL, vae=VAEFile, shift=SHIFT, legacy=False):
        print("Loading tokenizers...")
        # NOTE: if you need a reference impl for a high performance CLIP tokenizer instead of just using the HF transformers one,
        # check https://github.com/Stability-AI/StableSwarmUI/blob/master/src/Utils/CliplikeTokenizer.cs
        # (T5 tokenizer is different though)
        self.tokenizer = SD3Tokenizer(legacy)
        print("Loading OpenCLIP bigG...")
        self.clip_g = ClipG(model)
        print("Loading OpenAI CLIP L...")
        self.clip_l = ClipL(model)
        print("Loading Google T5-v1-XXL...")
        self.t5xxl = T5XXL(model)
        print("Loading SD3 model...")
        self.sd3 = SD3(model, shift)
        print("Loading VAE model...", end='', flush=True)
        self.vae = VAE(vae or model)
        print("Done")

    def get_empty_latent(self, width, height, num_samples=1):
        print("Prep an empty latent...")
        return torch.ones(num_samples, 16, height // 8, width // 8, device="cpu") * 0.0609

    def get_sigmas(self, sampling, steps):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(sampling.sigma(ts))
        sigs += [0.0]
        return torch.FloatTensor(sigs)

    def get_noise(self, seed, latent):
        generator = torch.manual_seed(seed)
        #print(f"dtype = {latent.dtype}, layout = {latent.layout}, device = {latent.device}")
        return torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu").to(latent.dtype)

    def get_cond(self, prompt, num_samples=1):
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        return torch.cat([lg_out, t5_out], dim=-2).repeat(num_samples, 1, 1), torch.cat((l_pooled, g_pooled), dim=-1).repeat(num_samples, 1)

    def max_denoise(self, sigmas):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

    def fix_cond(self, cond):
        cond, pooled = (cond[0].half().cuda(), cond[1].half().cuda())
        return { "c_crossattn": cond, "y": pooled }

    def do_sampling(self, latent, seed, conditioning, neg_cond, steps, cfg_scale, denoise=1.0, callback=None) -> torch.Tensor:
        message = 'Sampling...'
        print(message, end='', flush=True)
        callback(latent, 3, False, message=message)
        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()
        noise = self.get_noise(seed, latent).cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()
        sigmas = sigmas[int(steps * (1 - denoise)):]
        conditioning = self.fix_cond(conditioning)
        neg_cond = self.fix_cond(neg_cond)
        extra_args = { "cond": conditioning, "uncond": neg_cond, "cond_scale": cfg_scale }
        noise_scaled = self.sd3.model.model_sampling.noise_scaling(sigmas[0], noise, latent, self.max_denoise(sigmas))
        latent = sample_euler(CFGDenoiser(self.sd3.model), noise_scaled, sigmas, extra_args=extra_args, callback=callback)
        
        self.sd3.model = self.sd3.model.cpu()
        print("Done")
        return latent

    def vae_encode(self, image, num_samples=1) -> torch.Tensor:
        print("Encoding image to latent...", end='', flush=True)
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(num_samples, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        print("Done")
        return latent

    def vae_decode(self, latent) -> Image.Image:
        print("Decoding latent to image...", end='', flush=True)
        latent = latent.cuda()
        self.vae.model = self.vae.model.cuda()
        images = self.vae.model.decode(latent)
        images = images.float()
        self.vae.model = self.vae.model.cpu()
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        image_list = list()
        for image in images:
            decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
            decoded_np = decoded_np.astype(np.uint8)
            image_list.append(Image.fromarray(decoded_np))
        print("Done")
        return image_list

    def decode_latent(self, image) -> Image.Image:
        image = image.float()
        images = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        image_list = list()
        for image in images:
            decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
            decoded_np = decoded_np.astype(np.uint8)
            image_list.append(Image.fromarray(decoded_np[:,:,:3]))
        return image_list

    def gen_image(self, prompt, negative_prompt, callback, num_samples=1, width=1024, height=1024, steps=28, cfg_scale=5, seed=1, init_image=None, denoise=0.6):
        
        if init_image:
            callback(None, 0, False, message='Denoising input image...')
            latent = SD3LatentFormat().process_in(self.vae_encode(init_image, num_samples))
            
        else:
            callback(None, 0, False, message='Generating empty latent image...')
            latent = self.get_empty_latent(width, height, num_samples)

        message = 'Encoding prompt...'
        callback(latent, 1, False, message=message)

        print(message, end='', flush=True)
        conditioning = self.get_cond(prompt, num_samples)
        print('Done')
        message = 'Encoding negative prompt...'
        print(message, end='', flush=True)
        callback(latent, 2, False, message=message)
        neg_cond = self.get_cond(negative_prompt, num_samples)
        print('Done')
        
        
        sampled_latent = self.do_sampling(latent, seed, conditioning, neg_cond, steps, cfg_scale, denoise if init_image else 1.0, callback)
        callback(sampled_latent)
        #image = self.vae_decode(sampled_latent)
        #return image

