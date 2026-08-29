# References

Facts that are costly to derive again. Read the relevant file before you
start work. Add a new file when you find a fact that you had to measure or
look up, and that the code does not already state.

Rules for this directory:

- One subject per file.
- Give the command that produced each number.
- Keep a wrong result, and mark it wrong. It stops repeated work.
- Do not copy what the code already says. Link to the code instead.

## Files

| File | Subject |
|---|---|
| [test-shapes.md](test-shapes.md) | The 14 Appendix 3.7 shapes, their cost, and what each one loads |
| [machine.md](machine.md) | Hardware limits, library versions, and the timing rules |
| [mlx-tensorops.md](mlx-tensorops.md) | Which MLX kernel to use at which shape. Measured. |
| [scoreboard.md](scoreboard.md) | CPU/MPS/MLX timing for every shape, and a provisional MFU. Generated. |

## Related documents outside this directory

| File | Subject |
|---|---|
| [../OPTIMIZATIONS.md](../OPTIMIZATIONS.md) | Every optimization tried, kept or reverted, with numbers |
| [../profiling/README.md](../profiling/README.md) | Instruments and Metal capture setup, and the platform quirks |
| [../profiling/WORKFLOW.md](../profiling/WORKFLOW.md) | How to use the profiler to find the next optimization |

## Code map

| File | Role |
|---|---|
| `torch_transformer_benchmark.py` | The baseline model, the MLX model, and the accuracy and timing harness |
| `bench_cases.py` | Deterministic input generation, shared by every backend |
| `test_backends.py` | Cross-backend comparison: CPU, MPS, MLX |
| `appendix_cases.py` | The 14 shapes as code, with per-case selection |
| `flops.py` | The FLOP model, and the matmul rates measured here. |
| `scoreboard.py` | The graded run: CPU/MPS/MLX timing for every shape. Writes `scoreboard.md` and appends `profiling/history.jsonl`. |
| `profiling/sdpa_dispatch.py` | Finds which `head_dim` values reach the fused SDPA kernel, and when a pad into that set pays. |
