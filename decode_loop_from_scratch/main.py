from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import Tensor
import time
import numpy as np

model_name: str = "Qwen/Qwen3-8B"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class GeneralLM:
    def __init__(
        self, 
        top_value: float | int = 1.0,
        temperature_value: float = 1.0
    ):
        self.top_value = top_value
        self.temperature_value = temperature_value

    def temperature_application(self, logits: Tensor) -> Tensor:
        
        if isinstance(self.top_value, float) and (1.0 > self.top_value > 0):
            logits: Tensor = self._top_p_modification(logits)
        elif isinstance(self.top_value, int) and self.top_value > 1:
            logits: Tensor = self._top_k_modification(logits)

        logits: Tensor = logits / self.temperature_value
        probs: Tensor = torch.softmax(logits, dim=-1)
        next_token: Tensor = torch.multinomial(probs, num_samples=1)
        return next_token
    
    def _top_k_modification(self, logits: Tensor) -> Tensor:
        top_k_values: Tensor = torch.topk(logits, top_k, dim=-1).values
        return torch.masked_fill(logits, logits <= top_k_values[:, -1], float("-inf"))


    def _top_p_modification(self, logits: Tensor) -> Tensor:
        sorted_values, sorted_indicies = torch.sort(logits, descending=True)
        probs: Tensor = torch.softmax(input=sorted_values, dim=-1)
        cumulative_probs: Tensor = torch.cumsum(probs, dim=-1)
        sorted_mask: Tensor = cumulative_probs > top_p

        sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
        sorted_mask[..., 0] = False

        original_mask: Tensor = torch.zeros_like(input=sorted_mask, dtype=torch.bool)
        original_mask.scatter_(-1, sorted_indicies, sorted_mask)
        return logits.masked_fill(original_mask, float("-inf"))


class LanguageModel(GeneralLM):
    def __init__(
        self, 
        input_prompt: str, 
        top_value: float | int, 
        temperature_value: float,
        use_cache: bool = True
    ) -> None:
        super().__init__(
            top_value=top_value,
            temperature_value=temperature_value    
        )
        self.input_prompt: str = input_prompt
        self.use_cache: bool = use_cache

        # internal variables
        self.logits = None
        self.past_key_values = None
        self.output_sequence: list = []
        self.next_input = None

        self.token_idx_sequence = []

        # performance metrics
        self.ttft: float = 0
        self.list_token_speeds: list[float] = []

        self.clock: int = time.CLOCK_MONOTONIC

    @torch.inference_mode()
    def _prefill(self, use_cache: bool):

        torch.cuda.synchronize()
        start: int = time.clock_gettime_ns(self.clock)

        data = tokenizer(self.input_prompt, return_tensors="pt").to(device) # -> (46, 123, 5325)

        model_out = model(data.input_ids, use_cache=use_cache) # -> k, q, v

        self.token_idx_sequence = data.input_ids

        self.logits = model_out.logits # logit -> 

        if self.use_cache:
            self.past_key_values = model_out.past_key_values # kv

        logits = self.logits[:, -1, :] # (Batch, Time, vocab_size)
        next_token = self.temperature_application(logits)
        
        output = tokenizer.decode(next_token)
    
        self.output_sequence = output
        
        torch.cuda.synchronize()
        end: int = time.clock_gettime_ns(self.clock)
        
        self.next_input = next_token
        self.ttft: float = (end - start) / 1e9

    @torch.inference_mode()
    def generate(self, tokens: int = 100, warmup_mode: bool = False) -> str:
        if self.logits is None:
            self._prefill(self.use_cache)
        
        for _ in range(tokens):
            torch.cuda.synchronize()
            start: int = time.clock_gettime_ns(self.clock)
            if self.use_cache:
                model_out = model(
                    self.next_input, # token id
                    past_key_values=self.past_key_values
                )
                self.past_key_values = model_out.past_key_values
            else:
                model_out = model(self.token_idx_sequence, use_cache=False)

            self.logits = model_out.logits
            logits = self.logits[:, -1, :]
            next_token = self.temperature_application(logits)

            output = tokenizer.decode(next_token)

            self.token_idx_sequence = torch.cat([self.token_idx_sequence, next_token], dim=-1)

            self.output_sequence += output
            
            self.next_input = next_token
            torch.cuda.synchronize()
            end: int = time.clock_gettime_ns(self.clock)
            self.list_token_speeds.append((end - start) / 1e9)

        filling_var = self.output_sequence

        if warmup_mode:
            # internal variables
            self.logits = None
            self.past_key_values = None
            self.output_sequence: list = []
            self.next_input = None

            self.token_idx_sequence = []

            # performance metrics
            self.ttft: float = 0
            self.list_token_speeds: list[float] = []

            self.clock: int = time.CLOCK_MONOTONIC

        return filling_var

    def get_metrics(self):
        print(f"TTFT: {self.ttft} s")
        print(f"Mean time per token: {np.mean(self.list_token_speeds)} s")
        print(f"Min time per token: {np.min(self.list_token_speeds)} s")
        print(f"Max time per token: {np.max(self.list_token_speeds)} s")
        print(f"Total time: {np.sum(self.list_token_speeds)} s")


input_prompt: str = """
Now, I am going to count to 10000. 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
"""
temperature_value: float = 0.8
top_k: int = 5
top_p: float = 0.7


kv_cache_model: LanguageModel = LanguageModel(
    input_prompt=input_prompt,
    temperature_value=temperature_value,
    top_value=top_k,
    use_cache=True,
)

print(f"Warmup run: \n {"".join(kv_cache_model.generate(tokens=1024, warmup_mode=True))}")

print(100 * "=")

print(f"KV cache enabled model output: \n {"".join(kv_cache_model.generate(tokens=1024))}")
print("Metrics:")
kv_cache_model.get_metrics()

# no_cache_model: LanguageModel = LanguageModel(
#     input_prompt=input_prompt,
#     temperature_value=temperature_value,
#     top_value=top_k,
#     use_cache=False,
# )

# print(f"Warmup run: \n {"".join(no_cache_model.generate(tokens=512, warmup_mode=True))}")

# print(100 * "=")


# print(f"KV cache disabled model output: \n {"".join(no_cache_model.generate(tokens=1024))}")
# print("Metrics:")
# no_cache_model.get_metrics()