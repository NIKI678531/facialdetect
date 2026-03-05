import logging
from transformers import AutoModel, AutoProcessor
import torch
import os

# 获取环境变量 CLIP_DIR 的内容


def load_siglip_model():
    # 加载模型和处理器
    # ckpt = "google/siglip2-giant-opt-patch16-384"
    ckpt = "google/siglip2-so400m-patch16-384"
    model = AutoModel.from_pretrained(ckpt, device_map="mps").eval()
    processor = AutoProcessor.from_pretrained(ckpt, use_fast=True)

    # pretrained='/data/model_weight/mobileclip_blt.pt')
    # tokenizer = mobileclip.get_tokenizer('mobileclip_b')

    # 将模型移动到GPU
    # device = torch.device("cpu")
    # model = model.to(device)

    # logging.info(f"SigLIP model and processor initialized on {device}")
    return model, processor
