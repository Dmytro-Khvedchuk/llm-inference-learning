from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation.streamers import BaseStreamer
import torch
from prompts import PROMPT_128, PROMPT_1024
import time
import json
import argparse

parser = argparse.ArgumentParser(description="what this script does")
parser.add_argument("--model", choices=["Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B"], required=True)
args = parser.parse_args()

if args.model == "Qwen/Qwen3-1.7B":
    MODEL: str = "Qwen/Qwen3-1.7B"
    MODEL_SIGNATURE: str = "Qwen3-1.7B"
    PROMPT: str = PROMPT_1024
    S: int = 1024
elif args.model == "Qwen/Qwen3-8B":
    MODEL: str = "Qwen/Qwen3-8B"
    MODEL_SIGNATURE: str = "Qwen3-8B"
    PROMPT: str = PROMPT_128
    S: int = 256

BATCH_SIZES: list[int] = [1, 4, 8, 32, 64]
runs_count: int = 5
DO_SAMPLE: bool = True

# === CODE ===


def config_to_dict(cls):
    return {
        k: v for k, v in vars(cls).items()
        if not k.startswith("__") and not callable(v)
    }

class ModelConfig():
    MODEL: str = MODEL
    MODEL_SIGNATURE: str = MODEL_SIGNATURE
    PROMPT: str = PROMPT_128
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "bfloat16"
    S: int = S
    top_k: int = 20
    top_p: float = 0.8
    temperature: float = 1.0
    do_sample: bool = DO_SAMPLE
    current_date: str = str(datetime.now())

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
precision: str = "bfloat16"

top_k: int = 20
top_p: float = 0.8
temperature: float = 1.0

with open(f"{MODEL_SIGNATURE}.jsonl", "a") as file:
    file.write(json.dumps(obj=config_to_dict(ModelConfig)) + "\n")

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL, dtype=precision
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL,
)


class TimeStreamer(BaseStreamer):
    def __init__(self, start: float):
        self.records: list[float] = []
        self.prev_stop: float = start

    def put(self, value) -> None:
        current_stop: float = time.perf_counter()
        self.records.append(current_stop - self.prev_stop)
        self.prev_stop = current_stop

    def end(self):
        pass


class HFBenchmark:
    def __init__(self):
        self.measures: list[list[float]] = []

    def _warmup_run(self, B: int):
        print("Running the warmup...")
        prompts = tokenizer(self._prompt_multiplier(prompt=PROMPT, B=B), return_tensors="pt").to(
            DEVICE
        )

        model.generate(
            prompts.input_ids,
            attention_mask=prompts.attention_mask,
            max_length=S + prompts.input_ids.shape[1],
            min_length=S + prompts.input_ids.shape[1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=DO_SAMPLE
        )

    @staticmethod
    def _prompt_multiplier(prompt: str, B: int):
        return [prompt] * B


    def measure(self, runs_count: int = 2, B: int = 1):
        self._warmup_run(B)

        for run in range(runs_count):
            print(f"Processing run: {run}:")

            prompts = tokenizer(self._prompt_multiplier(prompt=PROMPT, B=B), return_tensors="pt").to(
                DEVICE
            )

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            streamer: TimeStreamer = TimeStreamer(start)

            model.generate(
                prompts.input_ids,
                attention_mask=prompts.attention_mask,
                max_length=S + prompts.input_ids.shape[1],
                min_length=S + prompts.input_ids.shape[1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                streamer=streamer,
                do_sample=DO_SAMPLE
            )

            if DEVICE == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            time_passed: float = end - start

            average_latency_ms: float = sum(streamer.records[2:]) / len(streamer.records[2:]) * 1000
            throughtput: float = B * S / time_passed
            decode_only_throughtput: float = B * len(streamer.records[2:]) / sum(streamer.records[2:])

            print(f"seconds passed: {time_passed:.4f} seconds")
            print(f"TTFT: {streamer.records[1]:.4f} seconds")
            print(f"Average latency: {average_latency_ms:.4f}ms")
            print(f"Throughtput: {throughtput:.4f} tokens/second")
            print(f"Decode only throughtput: {decode_only_throughtput:.4f} tokens/second\n")

            self.measures.append(streamer.records[1:])

            run_signature: str = f"MODEL_{MODEL_SIGNATURE}_RUN_{run}_B_{B}"

            data: dict[str, list[float]] = {
                run_signature : streamer.records[1:]
            }

            with open(f"{MODEL_SIGNATURE}.jsonl", "a") as file:
                file.write(json.dumps(data) + "\n")

benchmark = HFBenchmark()

for batch_size in BATCH_SIZES:
    benchmark.measure(runs_count=runs_count, B=batch_size)
