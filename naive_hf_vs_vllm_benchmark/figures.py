"""
1. Money plot: per-step latency vs step index, HF 1.7B B=64 (climbing ~28→~50 ms) vs vLLM flat — the thesis in one image.
2. Latency vs B (both frameworks, both models) — shows HF's flat launch floor and vLLM hugging the physics floor. B axis reads best on log₂ (your points are 1, 4, 8, 32, 64).
3. Throughput vs B — the knee (or its absence, for 8B) your sheet predicted.
"""
import matplotlib.pyplot as plt
import json
import numpy as np

data = {}

with open("naive_hf_vs_vllm_benchmark/artifacts/Qwen3-1.7B.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if len(row) == 1:
            (key, payload), = row.items()
            data[key] = payload



data_vllm = {}

with open("naive_hf_vs_vllm_benchmark/artifacts/VLLM_Qwen3-1.7B.jsonl") as f:
    for line in f:
        row = json.loads(line)
        if len(row) == 1:
            (key, payload), = row.items()
            data_vllm[key] = payload

runs = np.arange(5)
batches = [1, 4, 8, 32, 64]
colors = {1: "C0", 4: "C1", 8: "C2", 32: "C3", 64: "C4"}

plt.figure(figsize=(7,5))

average_latency = []
average_latency_vllm = []

average_throughput = []
average_throughput_vllm = []

for batch in batches:
    # vllm runs
    common_vllm = 0

    # hf runs
    common = []

    vllm_throughput = 0

    for run in runs:
        common.append(data[f"MODEL_Qwen3-1.7B_RUN_{run}_B_{batch}"][1:])

        common_vllm += data_vllm[f"VLLM_MODEL_Qwen3-1.7B_RUN_{run}_B_{batch}"]["latency"]
        
        vllm_throughput += data_vllm[f"VLLM_MODEL_Qwen3-1.7B_RUN_{run}_B_{batch}"]["decode_only_throughtput"]

    stack = np.array(common)
    curve = np.mean(stack, axis=0)

    average_throughput.append(batch * len(curve[1:]) / sum(curve[1:]))
    average_throughput_vllm.append(vllm_throughput / len(runs))

    average_latency_vllm.append(common_vllm / len(runs))
    average_latency.append(np.mean(curve) * 1e3)

    plt.plot(np.arange(len(curve)), np.full(len(curve), common_vllm / len(runs)),
             color=colors[batch], linestyle="--", alpha=0.6, label="_nolegend_")

    plt.plot(np.arange(len(curve)), curve * 1e3, color=colors[batch], label=f"B={batch}")


plt.plot([], [], color="gray", linestyle="--", label="vLLM (dashed)")
plt.axhline(15.95, color="black", linestyle=":", linewidth=2, label="BW floor @ B=64")

plt.title("Per-step decode latency Qwen3-1.7B — HF eager vs vLLM")
plt.legend(loc="upper left", fontsize=9, framealpha=0.95)

plt.xlabel("Token index")
plt.ylabel("Time spent for generation (ms)")

plt.savefig("naive_hf_vs_vllm_benchmark/figures/Qwen3-1.7B_latency_per_token.png", dpi=300, bbox_inches="tight")
plt.close()


# ===
plt.figure(figsize=(7,5))

b = ["B=1", "B=4", "B=8", "B=32", "B=64"]
x = np.arange(len(b))
width=0.35

plt.bar(x + width/2, average_latency, width=0.35, label="HF", color="orange")
plt.bar(x - width/2, average_latency_vllm, width=0.35, label="VLLM", color="#30a2ff")
plt.xticks(x, b)

plt.xlabel("Batch size")
plt.ylabel("Mean time for processing (ms)")
plt.legend(loc="best")
plt.title("Mean latency comparison Qwen3-1.7B — HF eager vs vLLM")

plt.savefig("naive_hf_vs_vllm_benchmark/figures/Qwen3-1.7B_mean_latency_per_batch.png", dpi=300, bbox_inches="tight")
plt.close()


# ===
plt.figure(figsize=(7,5))

b = ["B=1", "B=4", "B=8", "B=32", "B=64"]
x = np.arange(len(b))
width=0.35

plt.bar(x + width/2, average_throughput, width=0.35, label="HF", color="orange")
plt.bar(x - width/2, average_throughput_vllm, width=0.35, label="VLLM", color="#30a2ff")
plt.xticks(x, b)

plt.xlabel("Batch size")
plt.ylabel("Throughput tok/sec")
plt.legend(loc="best")
plt.title("Decode-only Throughput comparison Qwen3-1.7B — HF eager vs vLLM")

plt.savefig("naive_hf_vs_vllm_benchmark/figures/Qwen3-1.7B_decode_only_throughput_per_batch.png", dpi=300, bbox_inches="tight")
plt.close()