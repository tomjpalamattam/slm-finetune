import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm
from contextlib import nullcontext
import os
from contextlib import nullcontext
import tiktoken
import os
import numpy as np
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt
from datasets import load_dataset

from model import TransformerGPT, TransformerConfig
from utilities import get_batch
from lora import merge_lora_into_base, inject_lora_into_transformer
from config import *

device =  "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'


config = TransformerConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1
)
model = TransformerGPT(config)

model.load_state_dict(torch.load("best_model_params_orig.pt"))

inject_lora_into_transformer(model, rank=4, alpha=16, device=device)

for name, param in model.named_parameters():
    if "A" not in name and "B" not in name:
        param.requires_grad = False


best_val_loss = float('inf')
train_loss_list, validation_loss_list = [], []

model = model.to(device)
model.train()  # LoRA training mode (base weights frozen)


# Optimizer only over LoRA parameters
lora_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_params, lr=3e-4, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()

# LoRA fine-tuning on validation dataset
for epoch in tqdm(range(max_iters)):
    X, y = get_batch("val") 
    X, y = X.to(device), y.to(device)
    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    
    
best_model_params_path = "best_lora_params.pt"

torch.save(
    {k: v for k, v in model.state_dict().items() if "A" in k or "B" in k},
    best_model_params_path
)

merge_lora_into_base(model)

torch.save(model.state_dict(), "best_model_params.pt")
