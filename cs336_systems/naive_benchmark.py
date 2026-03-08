import gc
from pathlib import Path
import sys
import random
from collections.abc import Iterable, Callable
from typing import Any
import secrets
import timeit

import pandas as pd
import torch
from torch import Tensor
from jaxtyping import Int64
import click

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

CUDA = "cuda"


class Bench:
    def __init__(
        self,
        warmup_runs: int,
        timed_runs: int,
        device: str = "cpu",
        rand_seed: int | None = None,
    ) -> None:
        """ """
        self.warmup_runs = warmup_runs
        self.timed_runs = timed_runs
        self.rand_seed = rand_seed if rand_seed is not None else secrets.randbits(64)
        self.device = device

        # for reproducible randomness
        if rand_seed:
            random.seed(rand_seed)
            torch.manual_seed(rand_seed)

    def gen_model_input_targets(
        self, batch_size: int, seq_len: int, vocab_size: int
    ) -> tuple[
        Int64[Tensor, "batch_size seq_len"], Int64[Tensor, "batch_size seq_len"]
    ]:
        """Generate random input and targets.

        While we can simply compute whatever quantity which Pytorch can derive
        model weights' gradient from (e.g. `model(x).mean()` as shown in lecture),
        I prefer to approximate the reality that practical backward pass entails
        compututation of value of a non-trivial loss function eg cross-entropy.
        """
        t_input = torch.randint(
            0,
            vocab_size,
            (batch_size, seq_len),
            device=self.device,
            dtype=torch.int64,
        )
        t_target = torch.empty_like(t_input, device=self.device, dtype=torch.int64)
        t_target[:, 0:-1] = t_input[:, 1:]
        t_target[:, -1] = torch.randint(
            0,
            vocab_size,
            (batch_size,),
            device=self.device,
            dtype=torch.int64,
        )
        return t_input, t_target

    def run(
        self,
        model,
        vocab_size: int,
        ctx_len: int,
        batch_size: int,
        backward: bool = False,
    ):
        """
        backward: Account for model backward ops for benchmarking or not.
        Exec warmup runs first then timed runs.

        https://docs.python.org/3/library/timeit.html#module-timeit

        > default_timer() measurements can be affected by other programs
        > running on the same machine, so the best thing to do when accurate
        > timing is necessary is to repeat the timing a few times and use the best time.
        """
        x, targets = self.gen_model_input_targets(
            batch_size=batch_size, seq_len=ctx_len, vocab_size=vocab_size
        )

        def fn_to_benchmark():
            """Wrap code to benchmark into a callable w/ 0 params to align w/ timeit interface."""
            pred = model(x)
            if backward:
                loss = cross_entropy(pred, targets)
                loss.backward()
            if self.device == CUDA and torch.cuda.is_available():
                torch.cuda.synchronize()

        for _ in range(self.warmup_runs):
            fn_to_benchmark()

        # Repeat executing timeit() for self.timed_runs times
        measured_durations = timeit.repeat(
            fn_to_benchmark, repeat=self.timed_runs, number=1
        )
        # compute statistical summary of measured durations
        t_measured_durations = torch.tensor(measured_durations, dtype=torch.float32)
        return t_measured_durations.mean().item(), t_measured_durations.std().item()

def hyperparam_space(
    arrange: Callable,
    d_model: Iterable[int],
    d_ff: Iterable[int],
    num_layers: Iterable[int],
    num_heads: Iterable[int],
    ctx_len: Iterable[int],
    # TODO other hyperparams
) -> Iterable[dict[str, Any]]:
    """Generate space of hyperparams to explore.

    Args:
        arrange: Logic to arrange given hyperparameter iterables.
            Commonly used ones include `zip` to combine value at the
            same position in given iterables), `itertools.product` for
            cartesian product of values in given iterables
    """
    iters = [
        to_dicts("d_model", d_model),
        to_dicts("d_ff", d_ff),
        to_dicts("num_layers", num_layers),
        to_dicts("num_heads", num_heads),
        to_dicts("context_length", ctx_len),
    ]
    # merge kwargs from different dict in ds together and yield
    for ds in arrange(*iters):
        yield {k: v for d in ds for k, v in d.items()}


def arrange_fn(
    d_model: Iterable[dict],
    d_ff: Iterable[dict],
    num_layers: Iterable[dict],
    num_heads: Iterable[dict],
    ctx_len: Iterable[dict],
) -> Iterable[dict]:
    """Combine hyperparams however I prefer.
    This currently implement the criteria instructed by the assignment.

    Item in each iterable is in form of { '<hyperparam_name>':  value }

    Implement this as a generator.

    Spec:
    Assemble model cfgs except context length by zipping them together.
        This yields model cfgs similar to that of table in sec 1.1.2
    Then for each model cfg, create combinations w/ all given context
        length values by taking cartesian product of the two.
    """
    for ctx_len_v in ctx_len:
        for vs in zip(d_model, d_ff, num_layers, num_heads):
            yield (*vs, ctx_len_v)


def to_dicts(key: str, ls: Iterable[Any]) -> Iterable[dict]:
    return [{key: v} for v in ls]

@click.command("naive-benchmark", context_settings={"show_default": True})
@click.argument(
    "out",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-w",
    "--warmup-runs",
    type=click.INT,
    required=True,
    default=5,
    help="# warmup runs to execute before executing timed runs",
)
@click.option(
    "-t",
    "--timed-runs",
    type=click.INT,
    required=True,
    default=10,
    help="# timed runs to execute",
)
@click.option(
    "-b",
    "--backward",
    is_flag=True,
    default=False,
    help="Account for backward pass in benchmarking or not",
)
@click.option(
    "-d",
    "--device",
    default="cpu",
    help="Compute device to use. Eg 'cpu' / 'cuda'",
)
@click.option(
    "-s",
    "--rand-seed",
    type=click.INT,
    default=None,
    help="RNG seed for reproducible randomness behavior in debugging.",
)
@click.option(
    "--mem-snapshot-fp",
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Path of file to save the pytorch-managed cuda memory usage snapshot. Take snapshot if this option is set",
)
def run(
    out: Path,
    warmup_runs: int,
    timed_runs: int,
    backward: bool,
    device: str,
    rand_seed: int,
    mem_snapshot_fp: Path | None,
):
    """Naive Transformer model execution benchmarking.

    Args:
        OUT Path of file to record benchmark results in csv format.

    Spec:
    One execution of this program shall cover benchmark runs w/ different configs.
        Each config leads to a model using the config and benchmark run against the model.
        IMO the assignment wants benchmark against each model x various ctx length.

    Need a human-readable, concise way to identify configs used for each benchmark run.
    Program shall output both raw benchmark data and corresponding visualization at the end.
    """
    bench = Bench(
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        device=device,
        rand_seed=rand_seed,
    )

    # fixed params
    vocab_size = 10000
    batch_size = 4
    rope_theta = 10000

    params_space = hyperparam_space(
        arrange_fn,
        d_model=[
            768,
            1024,
            1280,
            1600,
            #2560,
        ],
        d_ff=[
            3072,
            4096,
            5120,
            6400,
            #10240,
        ],
        num_layers=[
            12,
            24,
            36,
            48,
            #32,
        ],
        num_heads=[
            12,
            16,
            20,
            25,
            #32,
        ],
        ctx_len=[
            128,
            256,
            512,
            1024,
        ],
    )

    snapshot_cuda_mem_usage = mem_snapshot_fp is not None
    if snapshot_cuda_mem_usage:
        # Record usage of CUDA memory allocated and managed by Pytorch allocator
        # More see https://docs.pytorch.org/docs/stable/torch_cuda_memory.html#torch.cuda.memory._dump_snapshot
        # and https://pytorch.org/blog/understanding-gpu-memory-1/
        torch.cuda.memory._record_memory_history()

    bench_results = []
    for params in params_space:
        # parameter combo / case identifier
        legend = ",".join([f"{k}={v}" for k, v in params.items()])
        print(f'Start benchmarking case {legend}', file=sys.stderr)

        try:
            model = BasicsTransformerLM(
                vocab_size=vocab_size,
                rope_theta=rope_theta,
                **params,
            )
            print(f'{legend} - Model initiated in physical RAM', file=sys.stderr)
            # move module's parameters to specified device
            model.to(device=device)
            print(f'{legend} - Model moved to GPU', file=sys.stderr)
        except Exception as e:
            print(f"Error when initiating model or moving it to GPU: {e}", file=sys.stderr)
            if snapshot_cuda_mem_usage:
                torch.cuda.memory._dump_snapshot(mem_snapshot_fp)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            break

        # start benchmark runs
        bench_stats = None
        bench_err = None
        try:
            bench_stats = bench.run(
                model,
                vocab_size=vocab_size,
                batch_size=batch_size,
                ctx_len=params["context_length"],
                backward=backward,
            )
        except Exception as e:
            print(f"Error when benchmarking case {legend}: {e}", file=sys.stderr)
            bench_err = str(e)

        if bench_stats:
            mean, std = bench_stats
            bench_results.append(params | {"bench_mean": mean, "bench_std": std})
        else:
            bench_results.append(params | {"err": bench_err})

        # attempt memory cleanup (currently not sure if this can work)
        # per https://stackoverflow.com/a/77587691
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Done benchmarking case {legend}', file=sys.stderr)

    if snapshot_cuda_mem_usage:
        torch.cuda.memory._dump_snapshot(mem_snapshot_fp)
    # Collect benchmark data into a Pandas Dataframe for easy recording, analysis
    # and future interactive visualization.
    bench_results_df = pd.DataFrame(bench_results)
    bench_results_df.to_csv(out, index=False)
    print(bench_results_df.to_markdown(index=False, tablefmt="github"))


if __name__ == "__main__":
    run()
