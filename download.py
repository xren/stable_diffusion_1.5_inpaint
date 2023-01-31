import torch
from diffusers import StableDiffusionInpaintPipeline
import os

inpainting_model_path = "runwayml/stable-diffusion-inpainting"


def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        inpainting_model_path,
        torch_dtype=torch.float16,
        use_auth_token=HF_AUTH_TOKEN
    )


if __name__ == "__main__":
    download_model()
