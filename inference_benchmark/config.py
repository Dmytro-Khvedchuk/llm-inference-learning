import torch

MODEL_NAME: str = "Qwen/Qwen3-8B"
MODEL_DTYPE: str = "bfloat16"
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu" 

PROMPT: str = "I'm a senior project manager. Create me roadmap for my career growth."