import os
import random
import math
import argparse

from tqdm import tqdm

import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

import numpy as np

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
    EulerAncestralDiscreteScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

TEMPLATE = (
    "<|bos|>"
    #
    "<|rating:general|>"
    "{aspect_ratio}"
    "<|length:medium|>"
    #
    "<copyright></copyright>"
    #
    "<character></character>"
    #
    "<general>{subject}<|input_end|>"
)

NEGATIVE_PROMPT = "bad quality, worst quality, lowres, bad anatomy, sketch, jpeg artifacts, ugly, poorly drawn, signature, watermark, bad anatomy, bad hands, bad feet, retro, old, 2000s, 2010s, 2011s, 2012s, 2013s, multiple views, screencap, anime coloring"


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_model",
        type=str,
        default="OnomaAIResearch/Illustrious-xl-early-release-v0",
    )
    parser.add_argument(
        "--dart",
        type=str,
        default="p1atdev/dart-v3-llama-7L2KV-241027_241028-sft-1",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=6.5,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.00,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
    )
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
    )
    parser.add_argument(
        "condition",
        type=str,
    )

    return parser.parse_args()


def prepare_dart(repo_id: str):
    dart = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map="cpu",
    )
    dart = dart.eval()
    dart = dart.requires_grad_(False)
    dart = torch.compile(dart)
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    return dart, tokenizer


def get_aspect_ratio(width: int, height: int) -> str:
    ar = math.log2(width / height)

    if ar <= -1.25:
        return "<|aspect_ratio:too_tall|>"
    elif ar <= -0.75:
        return "<|aspect_ratio:tall_wallpaper|>"
    elif ar <= -0.25:
        return "<|aspect_ratio:tall|>"
    elif ar < 0.25:
        return "<|aspect_ratio:square|>"
    elif ar < 0.75:
        return "<|aspect_ratio:wide|>"
    elif ar < 1.25:
        return "<|aspect_ratio:wide_wallpaper|>"
    else:
        return "<|aspect_ratio:too_wide|>"


def prepare_pipe(
    repo_id: str,
    device: str,
    cpu_offload: bool,
):
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        repo_id,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False,
        custom_pipeline="lpw_stable_diffusion_xl",
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if cpu_offload:  # local
        pipe.enable_sequential_cpu_offload(gpu_id=0, device=device)
    else:
        pipe.to(device)  # for spaces
    return pipe


@torch.inference_mode
def generate_prompt(
    dart,
    tokenizer,
    subject: str,
    aspect_ratio: str,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
):
    input_ids = tokenizer.encode_plus(
        TEMPLATE.format(aspect_ratio=aspect_ratio, subject=subject),
        return_tensors="pt",
    ).input_ids.repeat(batch_size, 1)
    # print("input_ids:", input_ids.shape)

    output_ids = dart.generate(
        input_ids.to(dart.device),
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=1,
        # bad_words_ids=BAN_TOKENS,
    )

    generated = output_ids[:, len(input_ids) :]
    decoded = [
        ", ".join(
            [
                token
                for token in tokenizer.batch_decode(sentence, skip_special_tokens=True)
                if token.strip() != ""
            ]
        )
        for sentence in generated.tolist()
    ]
    # print("decoded:", decoded)

    return decoded


@torch.inference_mode
def generate_image(
    pipe,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
):
    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
    ).images

    return images


def main():
    args = prepare_args()

    temperature = args.temperature
    top_p = args.top_p
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    condition = args.condition
    num_images = args.num_images
    cpu_offload = args.cpu_offload
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dart, tokenizer = prepare_dart(args.dart)
    dart.to(device)

    set_seed(args.seed)

    prompts = []

    chunks = np.array_split(range(num_images), num_images // batch_size)
    for chunk in tqdm(chunks):
        prompts.extend(
            generate_prompt(
                dart,
                tokenizer,
                condition,
                get_aspect_ratio(args.width, args.height),
                len(chunk),
                temperature,
                top_p,
                top_k,
                max_new_tokens,
            )
        )
    print("prompts:", len(prompts))
    del dart, tokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    pipe = prepare_pipe(args.image_model, device, cpu_offload=cpu_offload)
    for i, prompt in tqdm(enumerate(prompts)):
        images = generate_image(
            pipe,
            prompt,
            NEGATIVE_PROMPT,
            args.width,
            args.height,
            args.cfg,
            args.num_steps,
        )
        # print("images:", len(images))
        for image in images:
            image.save(f"{args.output_dir}/{i}.png")
            with open(f"{args.output_dir}/{i}.txt", "w") as f:
                f.write(prompt)


if __name__ == "__main__":
    main()
