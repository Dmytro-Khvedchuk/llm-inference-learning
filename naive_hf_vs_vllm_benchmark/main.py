from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import Tensor
import time
import numpy as np

model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
