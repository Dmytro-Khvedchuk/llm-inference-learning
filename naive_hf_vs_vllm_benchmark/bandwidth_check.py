import torch

N = 2 * 1024**3  # 2 GiB per tensor, well over cache sizes
x = torch.empty(N, dtype=torch.uint8, device="mps")
y = torch.empty(N, dtype=torch.uint8, device="mps")

for _ in range(3):          # warmup
    y.copy_(x)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
reps = 20
start.record()
for _ in range(reps):
    y.copy_(x)
end.record()
torch.cuda.synchronize()

seconds = start.elapsed_time(end) / 1000
gb_moved = reps * 2 * N / 1e9   # each copy READS 2 GiB and WRITES 2 GiB
print(f"{gb_moved / seconds:.1f} GB/s")