from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import Tensor
import time
import numpy as np

model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)
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

        # performance metrics
        self.ttft: float = 0
        self.list_token_speeds: list[float] = []

        self.clock: int = time.CLOCK_MONOTONIC

    def _prefill(self):
        start: int = time.clock_gettime_ns(self.clock)

        data = tokenizer(self.input_prompt, return_tensors="pt") # -> (46, 123, 5325)

        model_out = model(data.input_ids) # -> k, q, v

        self.logits = model_out.logits # logit -> 

        if self.use_cache:
            self.past_key_values = model_out.past_key_values # kv

        logits = self.logits[:, -1, :] # (Batch, Time, vocab_size)
        next_token = self.temperature_application(logits)
        output = tokenizer.decode(next_token)

        self.next_input = next_token
        self.output_sequence = output
        
        end: int = time.clock_gettime_ns(self.clock)
        self.ttft: float = (end - start) / 1e9

    def generate(self, tokens: int = 100) -> str:
        if self.logits is None:
            self._prefill()

        
        for _ in range(tokens):
            start: int = time.clock_gettime_ns(self.clock)
            if self.use_cache:
                model_out = model(
                    self.next_input, # token id
                    past_key_values=self.past_key_values
                )
                self.past_key_values = model_out.past_key_values
            else:
                data = tokenizer(
                    self.input_prompt + "".join(self.output_sequence), 
                    return_tensors="pt"
                )
                model_out = model(data.input_ids)

            self.logits = model_out.logits
            logits = self.logits[:, -1, :]
            next_token = self.temperature_application(logits)

            output = tokenizer.decode(next_token)
            self.output_sequence += output
            self.next_input = next_token
            
            end: int = time.clock_gettime_ns(self.clock)
            self.list_token_speeds.append((end - start) / 1e9)
            if next_token == tokenizer.eos_token_id:
                break

        return self.output_sequence

    def get_metrics(self):
        print(f"TTFT: {self.ttft} s")
        print(f"Mean time per token: {np.mean(self.list_token_speeds)} s")
        print(f"Min time per token: {np.min(self.list_token_speeds)} s")
        print(f"Max time per token: {np.max(self.list_token_speeds)} s")
        print(f"Total time: {np.sum(self.list_token_speeds)} s")


input_prompt: str = "Here is the citation from the "
temperature_value: float = 0.8
top_k: int = 5
top_p: float = 0.7


kv_cache_model: LanguageModel = LanguageModel(
    input_prompt=input_prompt,
    temperature_value=temperature_value,
    top_value=top_k,
    use_cache=True,
)

no_cache_model: LanguageModel = LanguageModel(
    input_prompt=input_prompt,
    temperature_value=temperature_value,
    top_value=top_k,
    use_cache=False,
)

print(f"KV cache enabled model output: \n {"".join(kv_cache_model.generate(tokens=50))}")
print("Metrics:")
kv_cache_model.get_metrics()

print(100 * "=")

print(f"KV cache disabled model output: \n {"".join(no_cache_model.generate(tokens=50))}")
print("Metrics:")
no_cache_model.get_metrics()