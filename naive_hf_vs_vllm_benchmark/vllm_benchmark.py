from vllm import LLM, SamplingParams
import argparse
from prompts import PROMPT_128, PROMPT_1024
import torch
import time
from datetime import datetime
import json

parser = argparse.ArgumentParser(description="what this script does")
parser.add_argument("--model", choices=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"], required=True)
args = parser.parse_args()

if args.model == "Qwen/Qwen3-1.7B":
    MODEL: str = "Qwen/Qwen3-1.7B"
    MODEL_SIGNATURE: str = "Qwen3-1.7B"
    PROMPT: str = PROMPT_1024
    S: int = 1024
    INPUT_S: int = 1024
elif args.model == "Qwen/Qwen3-8B":
    MODEL: str = "Qwen/Qwen3-8B"
    MODEL_SIGNATURE: str = "Qwen3-8B"
    PROMPT: str = PROMPT_128
    S: int = 256
    INPUT_S: int = 128

BATCH_SIZES: list[int] = [1, 4, 8, 32, 64]
runs_count: int = 5

# === CODE ===

def prompt_multiplier(prompt: str, B: int):
    return [prompt] * B

def config_to_dict(cls):
    return {
        k: v for k, v in vars(cls).items()
        if not k.startswith("__") and not callable(v)
    }

class ModelConfig():
    MODEL: str = MODEL
    MODEL_SIGNATURE: str = MODEL_SIGNATURE
    PROMPT: str = PROMPT
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "bfloat16"
    S: int = S
    INPUT_S: int = INPUT_S
    top_k: int = 20
    top_p: float = 0.8
    temperature: float = 1.0
    current_date: str = str(datetime.now()) 
    seed: int = 22
    enable_prefix_caching: bool = False
    gpu_memory_utilization=0.9

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
precision: str = "bfloat16"

top_k: int = 20
top_p: float = 0.8
temperature: float = 1.0

with open(f"VLLM_{MODEL_SIGNATURE}.jsonl", "a") as file:
    file.write(json.dumps(obj=config_to_dict(ModelConfig)) + "\n")


sampling_params = SamplingParams(
    top_k=top_k,
    top_p=top_p,
    temperature=temperature,
    seed=22,
    max_tokens=S,
    min_tokens=S,
)

sampling_params_ttft = SamplingParams(
    top_k=top_k,
    top_p=top_p,
    temperature=temperature,
    seed=22,
    max_tokens=1,
    min_tokens=1,
)

llm = LLM(
    model=MODEL,
    seed=22,
    gpu_memory_utilization=0.9,
    dtype=precision,
    enable_prefix_caching=False,
    max_model_len=S + INPUT_S
)

class VLLMBenchmark():
    


    def _warmup(self, batch_size: int):
        prompts: list = prompt_multiplier(PROMPT, B=batch_size)
        print("Warming up...")
        llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )

    def measure_ttft(self, batch_size: int):

        start: float = time.perf_counter()

        llm.generate(
            prompts=prompt_multiplier(PROMPT, B=batch_size),
            sampling_params=sampling_params_ttft,
        )

        end: float = time.perf_counter()

        ttft = end - start

        return ttft


    def measure(self, batch_size: int):
        self._warmup(batch_size)

        for run in range(runs_count):
            print(f"Processing run: {run} for batch_size: {batch_size}")

            ttft = self.measure_ttft(batch_size)

            start: float = time.perf_counter()

            outputs = llm.generate(
                prompts=prompt_multiplier(PROMPT, B=batch_size),
                sampling_params=sampling_params,
            )

            end: float = time.perf_counter()


            time_passed: float = end - start

            latency: float = (time_passed - ttft) / (int(sampling_params.max_tokens) - 1) * 1000
            decode_only_throughtput: float = batch_size / latency * 1000

            throughtput: float = batch_size * sampling_params.max_tokens / time_passed


            metrics: dict ={
                "latency": latency,
                "decode_only_throughtput": decode_only_throughtput,
                "throughtput": throughtput,
                "TTFT": ttft,
                "time_passed": time_passed,
                "num_cached_tokens": outputs[batch_size - 1].num_cached_tokens,
                "prompt_len": len(outputs[batch_size - 1].prompt_token_ids)
            }

            print(f"Seconds passed for the completion: {time_passed:.4f}")
            print(f"Latency: {latency:.4f}ms")
            print(f"Throughtput decode_only: {decode_only_throughtput:.4f} tokens/second")
            print(f"Throughtput: {throughtput:.4f} tokens/second")

            run_signature: str = f"VLLM_MODEL_{MODEL_SIGNATURE}_RUN_{run}_B_{batch_size}"

            data: dict[str, dict] = {
                run_signature : metrics
            }

            with open(f"VLLM_{MODEL_SIGNATURE}.jsonl", "a") as file:
                file.write(json.dumps(data) + "\n")

benchmark: VLLMBenchmark = VLLMBenchmark()

for batch_size in BATCH_SIZES:
    benchmark.measure(batch_size)
