#!/bin/bash

cd /home/tom/

export GRADIO_TEMP_DIR="/home/tom/fssd/tmp/"
export HUGGINGFACE_HUB_CACHE="/home/tom/fssd/HF_cache"
export HF_HOME="/home/tom/fssd/HF"

export MODELS_DIR="/home/tom/fssd/models/ckpt/"
export FLUX_DEV="/home/tom/fssd/models/ckpt/flux1-dev.safetensors"
export FLUX_DEV_FP8="/home/tom/fssd/models/ckpt/flux-dev-fp8.safetensors"
export AE="/home/tom/models/ae.safetensors"
export TEXT_ENCODER="/home/tom/models/xflux_text_encoders"
export CLIP="/home/tom/models/clip-vit-large-patch14"

export HF_HUB_ENABLE_HF_TRANSFER=1

cd PuLID && python launch_app.py && cd /home/tom/