
from pathlib import Path
import random
import argparse
import datetime
import base64
import io
from PIL import Image

import torch
import numpy as np

from src.sd3_infer import SD3Inferencer
from src.sd3_impls import SD3LatentFormat


from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "stable_diffusion_3"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d7"
VERSION = 0


class ProcessOutputCallback():
    def __init__(self, api_worker, inferencer):
        self.api_worker = api_worker
        self.inferencer = inferencer
        self.job_data = None


    def process_output(self, latent_image, progress=100, finished=True, error=None):
        
        if error:
            print('error')
            self.api_worker.send_progress(100, None)
            image = Image.fromarray((np.random.rand(1024,1024,3) * 255).astype(np.uint8))
            return self.api_worker.send_job_results({'images': [image], 'error': error})
        else:
            if not finished:
                if self.api_worker.progress_data_received:
                    if self.job_data.get('provide_progress_images') == 'None':
                        return self.api_worker.send_progress(progress)
                    elif self.job_data.get('provide_progress_images') == 'decoded':
                        image_list = self.inferencer.vae_decode(SD3LatentFormat().process_out(latent_image))
                    elif self.job_data.get('provide_progress_images') == 'latent':
                        image_list = SD3LatentFormat().decode_latent_to_preview(latent_image)
 
                    return self.api_worker.send_progress(progress, {'progress_images': image_list})
            else:
                image_list = self.inferencer.vae_decode(SD3LatentFormat().process_out(latent_image))
                self.api_worker.send_progress(100, None)
                return self.api_worker.send_job_results({'images': image_list})


    def get_image_list(self, images):
        image_list = list()
        for image in images:
            image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
            image = Image.fromarray(image.astype(np.uint8))
            image_list.append(image)
        return image_list


def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the AIME API server"
                        )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used"
                        )
    parser.add_argument(
        "--ckpt_dir", type=str, default="/data/models/stable-diffusion-3-medium/", help="Destination of model weigths"
                        )
    parser.add_argument(
        "--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, 
        help="API server worker auth key",
    )
    parser.add_argument(
        "--legacy", action='store_true', help="Use T5 Tokenizer legacy behaviour, see https://github.com/huggingface/transformers/pull/24565 for more info"
    )

    return parser.parse_args()


def convert_base64_string_to_image(base64_string, width, height):
    if base64_string:
        base64_data = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_data)

        with io.BytesIO(image_data) as buffer:
            image = Image.open(buffer)
            return image.resize((width, height), Image.LANCZOS)


def set_seed(job_data):
    
    seed = job_data.get('seed', -1)
    if seed == -1:
        random.seed(datetime.datetime.now().timestamp())
        seed = random.randint(1, 99999999)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    job_data['seed'] = seed
    return job_data


@torch.no_grad()
def main():
    args = load_flags()
    torch.cuda.set_device(args.gpu_id)
    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, args.api_auth_key, args.gpu_id, world_size=1, rank=0, gpu_name=torch.cuda.get_device_name())
    inferencer = SD3Inferencer()
    inferencer.load(Path(args.ckpt_dir), None, 3.0, args.legacy)
    callback = ProcessOutputCallback(api_worker, inferencer)

    while True:
        try:
            job_data = api_worker.job_request()
            print(f'Processing job {job_data.get("job_id")}...', end='', flush=True)
            job_data = set_seed(job_data)
            init_image = job_data.get('image')
            if init_image:
                init_image = convert_base64_string_to_image(
                    init_image,
                    job_data.get('width'), 
                    job_data.get('height')
                )
            callback.job_data = job_data           
            image = inferencer.gen_image(
                job_data.get('prompt'),
                job_data.get('negative_prompt'),
                callback.process_output,
                job_data.get('num_samples', 1),
                job_data.get('width'), 
                job_data.get('height'), 
                job_data.get('steps'), 
                job_data.get('cfg_scale'), 
                job_data.get('seed'),
                init_image,
                job_data.get('denoise')
            )
            print('Done')
        except ValueError as exc:
            print('Error')
            callback.process_output(None , 100, True, f'{exc}\nChange parameters and try again')
            continue
        except torch.cuda.OutOfMemoryError as exc:
            print('Error')
            callback.process_output(None, 100, True, f'{exc}\nReduce number of samples or image size and try again')
            continue
        except OSError as exc:
            print('Error')
            callback.process_output(None, 100, True, f'{exc}\nInvalid image file')
            continue

if __name__ == "__main__":
    main()