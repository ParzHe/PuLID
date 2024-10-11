import os
import torch

from pulid.utils import get_gpu_mem_info

if __name__ == "__main__":
    
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
    cmd = ""
    
    torch.cuda.empty_cache()
    
    models_dir = os.getenv("MODELS_DIR")
    if gpu_mem_total > 30:
        if not os.path.isfile(str(os.getenv("FlUX_DEV"))):
            print("第一次启动需要拷贝约22G模型文件到fssd中")
            os.makedirs(models_dir,exist_ok=True)
            os.system(f"cp -v -n /home/tom/fshare/models/black-forest-labs/FLUX.1-dev/flux1-dev.safetensors {models_dir}")
        if gpu_mem_total > 45:
            cmd="python app_flux.py"
        else:
            cmd="python app_flux.py --offload"
    else: 
        print(f"所选显卡显存总共为{gpu_mem_total} GB。其不足以载入非量化的Flux-dev 自动使用 FP8 版本...")
        if not os.path.isfile(str(os.getenv("FlUX_DEV_FP8"))):
            print("第一次启动需要拷贝约11G模型文件到fssd中")
            os.makedirs(models_dir,exist_ok=True)
            os.system(f"cp -v -n /home/tom/fshare/models/XLabs-AI/flux-dev-fp8/flux-dev-fp8.safetensors {models_dir}")
        cmd="python app_flux.py --offload --fp8"
    
    os.system(cmd)