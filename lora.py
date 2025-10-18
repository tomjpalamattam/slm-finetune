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

from model import MultiHeadAttention

class LoRALinear(nn.Module):
    """
    LoRA wrapper for a single nn.Linear layer.
    It adds a low-rank residual (A @ B) * scaling to the output.
    """

    def __init__(self, layer: nn.Linear, rank: int = 4, alpha: float = 1.0, device=None):
        super().__init__()
        self.layer = layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.in_features = layer.in_features
        self.out_features = layer.out_features

        # LoRA parameters (small)
        self.A = nn.Parameter(torch.zeros(rank, self.in_features, device=device))
        self.B = nn.Parameter(torch.zeros(self.out_features, rank, device=device))
        self.reset_parameters()

        # Freeze the original layer
        for p in self.layer.parameters():
            p.requires_grad = False

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # Normal forward + LoRA residual
        result = self.layer(x)
        lora_update = (x @ self.A.T @ self.B.T) * self.scaling
        return result + lora_update
    
    
def inject_lora_into_transformer(model, rank=4, alpha=1.0, device=None):
    """
    Replaces each MultiHeadAttention's linear layers with LoRALinear-wrapped versions.
    """
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            module.W_q = LoRALinear(module.W_q, rank=rank, alpha=alpha, device=device)
            module.W_k = LoRALinear(module.W_k, rank=rank, alpha=alpha, device=device)
            module.W_v = LoRALinear(module.W_v, rank=rank, alpha=alpha, device=device)
            # Optional:
            # module.W_o = LoRALinear(module.W_o, rank=rank, alpha=alpha, device=device)
            print(f"Injected LoRA into MultiHeadAttention layer: {name}")
            
            
def merge_lora_into_base(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                W = module.layer.weight.data
                delta = (module.B @ module.A) * module.scaling
                W += delta.clone()           
