from vllm import LLM, SamplingParams, RequestOutput

from config import MODEL_NAME, MODEL_DTYPE, DEVICE, PROMPT

sampling_params: SamplingParams = SamplingParams(
    n=2,    
    temperature=1,
    top_k=5,
    seed=22,
    min_tokens=256,
    max_tokens=1024
)

llm: LLM = LLM(
    model=MODEL_NAME,
    dtype=MODEL_DTYPE,
    seed=22,
    gpu_memory_utilization=0.90,
    max_model_len=4096
)

outputs: list[RequestOutput] = llm.generate(
    prompts=PROMPT,
    sampling_params=sampling_params
)


ind = 0
for out in outputs:
    print(out.outputs[ind])
    print("==" * 50)
    ind+=1
    