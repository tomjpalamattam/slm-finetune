## TransformerGPT with LoRA Fine-Tuning

This repository implements a **Transformer-based GPT model** from scratch using **PyTorch**, with support for **LoRA (Low-Rank Adaptation)** fine-tuning for efficient parameter updates.
It includes full data preprocessing, model training, LoRA injection, and text generation pipelines.

---

## Repository Structure

```
.
├── config.py          # Configuration dataclass for Transformer hyperparameters
├── model.py           # Core TransformerGPT architecture (attention, feedforward, embeddings, etc.)
├── lora.py            # LoRA linear wrapper and injection utilities
├── train.py           # Base model training script (full fine-tuning)
├── lora_train.py      # LoRA fine-tuning script (parameter-efficient)
├── utilities.py       # Data processing, batching, loss estimation, schedulers, etc.
├── Notebook/          # Jupyter notebooks for visualization or experimentation
└── README.md         
```

---

## ⚙️ Features

* **Transformer architecture** implemented from scratch:

  * Multi-head self-attention
  * Positional encoding
  * Layer normalization and residual connections
  * Feed-forward layers

* **LoRA fine-tuning support**:

  * Efficient adaptation by injecting low-rank adapters into attention layers
  * Reduces trainable parameters drastically (from ~49M → ~55K)

* **GPT-style token generation** with causal masking

* **Mixed-precision training** 

* **Visualization of training losses and computation graph**

---

## 🧩 Requirements

Create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install datasets tiktoken tqdm matplotlib torchviz
```

---

## 📊 Dataset

The code uses the **TinyStories** dataset:

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
```

---

## 🚀 Training the Model


```bash
python train.py
```

### Fine-Tune with LoRA

Once the base model is trained, inject LoRA adapters and fine-tune efficiently:

```bash
python lora_train.py
```

This will:

* Inject LoRA modules into all attention layers
* Freeze the base model weights
* Train only LoRA parameters (~55k params)
* Merge LoRA updates into the base model
* Save the merged model as `best_model_params.pt`

---

## 🧠 Text Generation

After training or fine-tuning, you can generate text like this:

```python
import torch
import tiktoken
from model import TransformerGPT
from config import TransformerConfig

# Load tokenizer and model
enc = tiktoken.get_encoding("gpt2")
config = TransformerConfig(vocab_size=50257, block_size=128, n_layer=6, n_head=6, n_embd=384)
model = TransformerGPT(config)
model.load_state_dict(torch.load("best_model_params.pt", map_location="cpu"))
model.eval()

sentence = "There lived a girl"
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0)
generated = model.generate(context, max_new_tokens=200)
print(enc.decode(generated.squeeze().tolist()))
```

---


## 🧪 LoRA Stats

| Metric               | Before LoRA | After LoRA |
| :------------------- | ----------: | ---------: |
| Trainable Parameters |  49,244,928 |     55,296 |
| Total Parameters     |  49,244,928 | 49,300,224 |


---

## 📝 References

* [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
* [PyTorch AMP documentation](https://pytorch.org/docs/stable/amp.html)
* [HuggingFace Datasets](https://huggingface.co/docs/datasets)

---

