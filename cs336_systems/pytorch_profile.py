"""Profile Transformer model GPU memory usage by categories e.g.
model params, activations, gradients etc.

NOTE this is using a deprecated profiling methodology. Keep this only for recording purpose.

Spec:
    Prepare parameters to initiate Transformer model and run its forward / backwrad pass.
    Set up pytorch profiler per:
        - https://docs.pytorch.org/docs/main/profiler.html#torch.profiler.profile
        - https://pytorch.org/blog/understanding-gpu-memory-1/
    Record mem and flops utilization of model's execution w/ profiler
"""

import torch
from cs336_basics.transformer import TransformerModel
from cs336_systems.naive_benchmark import CUDA, Bench, arrange_fn


if __name__ == "__main__":
    assert torch.cuda.is_available(), 'Current profiling logic requires cuda being available'
    # fixed params
    vocab_size = 10000
    batch_size = 4
    rope_theta = 10000
    context_len = 128

    model = TransformerModel(
        vocab_size=vocab_size,
        rope_theta=rope_theta,
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        context_len=context_len,
        device=CUDA,
        dtype=torch.float32,
    )

    bench = Bench(warmup_runs=0, timed_runs=0, device=CUDA, rand_seed=102442)
    x, targets = bench.gen_model_input_targets(
            batch_size=batch_size, seq_len=context_len, vocab_size=vocab_size)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0, warmup=0, active=6,
            repeat=0,
            ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as pt_prof:
        # examine profiler's behavior when # iter differs from schedule's # active steps
        for _ in range(5):
            _ = model(x)
            # TODO profile backward ops
            torch.cuda.synchronize()
            pt_prof.step()

    pt_prof.export_memory_timeline('pytorch-prof-mem-utilization.html', device="cuda:0")
    print(pt_prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))