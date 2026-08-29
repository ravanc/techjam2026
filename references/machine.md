# Machine and toolchain

Every measurement in this repository comes from this machine. Repeat a
measurement here before you trust a number from any other source.

## Hardware

| Item | Value |
|---|---|
| Chip | Apple M3 Pro |
| CPU | 6 performance cores + 5 efficiency cores |
| GPU | 14 cores |
| Unified memory | 18 GiB (19,327,352,832 bytes) |
| GPU max working set | 12.0 GiB (12,884,918,272 bytes) |
| GPU max buffer | 9.0 GiB (9,663,676,416 bytes) |
| GPU architecture | `applegpu_g15s` |

Read these values again with:

    .venv/bin/python3 -c "import mlx.core as mx; print(mx.device_info())"

The memory is unified. A torch CPU tensor and an MLX GPU array use the same
physical DRAM. A "transfer" is therefore a copy inside one memory, not a bus
transfer. This is why the framework boundary costs under 1% of a call.

**The 12.0 GiB working set is the hard limit.** It is not the 18 GiB of
system memory. Any shape that needs more than 12 GiB of live GPU arrays must
use a chunk loop.

## Software

| Item | Value |
|---|---|
| Python | 3.13.5, in `.venv` |
| torch | 2.13.0 |
| mlx | 0.32.2 |
| numpy | 2.5.2 |
| macOS | Darwin 24.6 |
| Xcode | 16.4, with the Metal toolchain |

torch has no CUDA on this machine. `torch.backends.mps.is_available()` is
True, so `mps` is the torch GPU device.

**Always use the venv.** Run `.venv/bin/python3`, never `python3`.

## Peak rates

Use these to tell a slow kernel from a shape that is simply small.

| Item | Rate |
|---|---|
| GPU float32 matmul, peak | about 3.5 TFLOP/s |
| Unified memory bandwidth | about 150 GB/s |

A kernel that reaches under 100 GFLOP/s is either launch-bound or
memory-bound. Check the arithmetic intensity before you rewrite it.

## Timing rules

These three mistakes give wrong numbers on this machine. All three have
already happened in this repository.

1. **MLX is lazy and asynchronous.** `mx.eval()` alone is not enough for a
   timing loop. Call `mx.synchronize()` after `mx.eval()`.
2. **torch MPS is asynchronous.** `benchmark_once()` in
   `torch_transformer_benchmark.py` synchronizes only on CUDA. It therefore
   times *enqueue* for a torch MPS baseline, and completion for the MLX path.
   Reported MPS speedups from that harness are not valid. See
   [../profiling/README.md](../profiling/README.md).
3. **A short run gives a false median.** `--repeats 3` gave 7.198x where
   `--repeats 100` gave 4.590x on the same build. Use the default repeats.
