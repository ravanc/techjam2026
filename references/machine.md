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
transfer.

**A copy is still not free.** An earlier version of this file said the
framework boundary costs under 1% of a call. That is wrong. `_to_mlx` copies
the 655 MiB shape 6 input on the CPU, and the GPU sits idle for **15.9 ms of a
462 ms call, which is 3.4%** (row 52, read off the Metal timeline). Row 48
measured the whole boundary at 5.4% of shape 6. Unified memory removes the bus,
not the bytes.

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

| Item | Rate | Source |
|---|---|---|
| GPU float32, theoretical peak | 4.946 TFLOP/s | `14 x 128 x 2 x 1.380 GHz`, see below |
| GPU float32 matmul, **measured** | **4.06 TFLOP/s** | `flops.py --peak` |
| GPU float32 pure FMA loop, measured | 3.92 TFLOP/s | `profiling/probes/alu_peak.py` |
| Memory bandwidth, specification | 150 GB/s | Apple, for the M3 Pro |
| Memory bandwidth, **measured streaming** | **128 GB/s** | `x * 2.0` at 1 GiB |

Use the measured 128 GB/s as the roof, not the 150 GB/s specification. A
kernel cannot exceed what a plain copy reaches. Use 4.06 TFLOP/s the same
way: it is the best rate anything has reached here.

### Where the 4.946 TFLOP/s comes from

    peak = cores x ALUs per core x 2 flop per FMA x clock
         = 14 x 128 x 2 x 1.380 GHz

| term | value | how it was checked |
|---|---|---|
| cores | 14 | `system_profiler SPDisplaysDataType`; also `gpu-core-count` in the `AGXAccelerator` IORegistry node |
| clock | 1.380 GHz | the GPU DVFS table, below |
| ALUs per core | 128 | bounded below at 105 by measurement, below |
| flop per FMA | 2 | one multiply and one add |

**An earlier version used 1.398 GHz, asserted from memory. It is wrong.**
The hardware publishes its own GPU DVFS table:

    ioreg -lw0 -p IODeviceTree -n pmgr | grep -o '"voltage-states9" = <[0-9a-f]*>'

It decodes as pairs of little-endian uint32, `{frequency Hz, millivolts}`:

| MHz | 0 | 338 | 618 | 796 | 832 | 924 | 952 | 1056 | 1064 | 1182 | 1182 | 1312 | 1242 | **1380** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mV | 125 | 665 | 705 | 710 | 745 | 745 | 805 | 805 | 880 | 880 | 935 | 935 | 965 | 965 |

`voltage-states9` is the GPU rail, not a CPU cluster: its top state is far
below any CPU core, its voltage ramps like a real rail, and it has a
`voltage-states9-sram` companion exactly as the two CPU clusters do.

**The ALU count is bounded, not read.** Nothing on this machine publishes it.
But a matmul reaches 4.06 TFLOP/s and the GPU cannot exceed 1380 MHz, so

    ALUs per core >= 4.06e12 / (14 x 2 x 1.380e9) = 105.1

64 is impossible, and 128 is the next width an Apple GPU core has.

### The GPU does not hold 1380 MHz

Both saturating measurements imply a clock near 1.1 GHz, not 1.38 GHz:

| what | rate | implied clock at 128 ALUs |
|---|---:|---:|
| pure FMA loop, sustained | 3.92 TFLOP/s | 1.09 GHz |
| float32 matmul | 4.06 TFLOP/s | 1.13 GHz |

1.09 GHz sits between the 1064 and 1182 MHz states of the table above, so
this is the DVFS state the chip chooses under a sustained load, not a
measurement error. **It is not a warm-up effect.** `alu_peak.py` queues 24
kernels into one `eval` and holds the GPU busy for 185 ms with no gap; the
rate rises from 3.86 to 3.92 TFLOP/s and stops there.

So an MFU against 4.946 TFLOP/s carries an 18% penalty that no kernel can
remove. Read the MFU column against 82%.

**A causal shape can print more than 82%, and shape 13 does.** The FLOP model
counts the FULL `S x S` attention, because that is what `BaselineTransformer`
computes, while the optimized path skips the upper triangle. So a long
sequence is credited with work it never runs:

| shape | counted GFLOP | executed GFLOP | printed MFU | MFU on executed work |
|---:|---:|---:|---:|---:|
| 13 (S=1024) | 188.98 | 120.33 | 91.9% | **58.5%** |
| 6 (S=128) | 1342.18 | 1175.72 | 54.5% | 47.7% |
| 8 (S=128) | 429.50 | 420.97 | 71.8% | 70.4% |

The 82% ceiling applies to executed work. Get the second column from
`flops.model_flops(config, causal_aware=True)`.


**Check for load that is not Python.** The rule 1 command in CLAUDE.md
greps for `.venv/bin/python3`, so it finds a competing sweep and nothing
else. It missed a `mysqld` holding 76% of a CPU, and that run came in at 5.3
minutes against 3.0 for the same sweep, with the MPS control 5% slow. Check
the whole machine:

    ps -Ao %cpu,%mem,command -r | head -6

Anything above a few percent that is not yours makes the reading false. The
MPS column is the detector: `BaselineTransformer` never changes, so if MPS
moved more than about 1%, the machine moved and the sweep cannot score a
change. A per-shape MPS control still can.

**Measure the stage the same way you measured the roof.** The 128 GB/s above
is a raw reading: it includes one `mx.eval` + `mx.synchronize` round trip.
`profiling/probes/stage_roofline.py` subtracts that round trip from its stage times
but not from this roof, so its `%mem` column overstates and can pass 100%.
Compare its `raw` column against this roof, not its `ms` column. Measured on
the shape 6 `ln1`, which moves 128 MiB: `ms` 0.9477 (141.6 GB/s, prints
110.6%) against `raw` 1.2492 (107.4 GB/s, the true figure). See
OPTIMIZATIONS.md row 43.

**An array under 64 MiB does not reach the roof.** Measured with `x * 2.0`,
reading plus writing:

| array | 1 MiB | 4 MiB | 16 MiB | 64 MiB | 256 MiB | 1 GiB |
|---|---:|---:|---:|---:|---:|---:|
| GB/s | 7.1 | 34.9 | 72.9 | 109.0 | 125.6 | 128.1 |

That table alone explains the small shapes. At shape 1 one activation is
4 MiB, so every data movement stage there runs at a quarter of the roof.

**Ridge point: 4.06e12 / 128e9 = 31.7 FLOP/byte.** A stage below that
intensity is memory-bound and cannot reach the arithmetic peak, whatever
the kernel does. Every projection in this model sits at 32 FLOP/byte, right
on the ridge, except the shape 8 projections at 241 to 351.

A kernel that reaches under 100 GFLOP/s is either launch-bound or
memory-bound. Check the arithmetic intensity before you rewrite it.

## The kernel launch floor

One `mx.eval()` + `mx.synchronize()` round trip costs **0.12 to 0.41 ms**,
and it includes the CPU graph build of one small operation. Extra kernels
inside one `eval` cost about **0.0016 ms** each, so the round trip dominates.

**An earlier version of this section said 0.004 ms for the extra kernel.
That is 2.5x too high.** Measured 31 August 2026 by a slope, not by a
single reading: queue K copies of a 1024-float elementwise operation into
one `eval`, sweep K over 1, 2, 4, 8, 16, 32, 64, and fit a line. The whole
sweep spans 0.1553 to 0.3211 ms, so 64 extra kernels cost less than one
round trip. The fitted slope is **1.55 us for each kernel**. The same
script gives the pure GPU cost of a real GEMM the same way, with no floor
to subtract:

| operation | slope, GPU ms per call | rate |
|---|---:|---:|
| elementwise, 1024 floats | 0.0016 | dispatch only |
| addmm 8192 x 128 x 128 | 0.0657 | 4.08 TFLOP/s |
| addmm 65536 x 128 x 128 | 0.6291 | 3.41 TFLOP/s |
| addmm 131072 x 128 x 128 | 1.2616 | 3.40 TFLOP/s |

So dispatch is 2.4% of the smallest real GEMM in the model and 0.12% of
the largest. **The forward pass launches 29 kernels for each chunk**, which
is 0.045 ms of dispatch: 0.10% of a shape 6 chunk.

**The floor is not a constant. It tracks the CPU load.** Three runs of
`stage_roofline.py` on 30 August 2026, minutes apart, measured 0.3049,
0.1468 and 0.3214 ms, while `mysqld` held 96% of one CPU. A run on 31
August 2026 measured 0.4122 ms with a screen saver holding the GPU, and
0.1250 ms once it stopped. The round trip is CPU-side work, so a busy
machine raises it. Measure it in the same run that uses it, and never
carry a floor from an earlier run.

Subtract this floor from any timing of a single operation. At shape 2 the
whole model is 0.75 ms, which is four round trips, so every stage of that
shape sits at the floor. Measure it again with:

    .venv/bin/python3 profiling/probes/stage_roofline.py --shapes 2

## mx.fast.layer_norm collapses below a row width of 256

`mx.fast.layer_norm` loses throughput as the normalized row gets narrower.
Nothing else does. Measured at a constant 64 MiB, so the row count rises as
`D` falls. GB/s counts the read plus the write:

| row width `D` | 32 | 64 | 96 | **128** | 192 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| copy `x * 2` | 108 | 109 | 108 | 109 | 108 | 106 | 108 | 110 | 110 |
| `fast.rms_norm` | 74 | 106 | 107 | **107** | 105 | 107 | 107 | 109 | 108 |
| `fast.layer_norm` | **5.5** | **12** | **21** | **33** | 87 | 105 | 107 | 107 | 106 |
| layer_norm / copy | 19.8x | 8.8x | 5.0x | **3.3x** | 1.2x | 1.0x | 1.0x | 1.0x | 1.0x |

At `D >= 256` layer_norm runs at copy speed. Below 256 it degrades roughly
as `1/D`, which is the signature of a kernel that gives one thread or one
small group to each row and does not vectorize along it.

**The penalty belongs to `layer_norm` alone.** The two pieces it is built
from both run at full speed at every width:

| operation at `D` = | 32 | 128 | 256 | 1024 |
|---|---:|---:|---:|---:|
| `mx.mean(x, axis=-1)`, GB/s of the read | 93 | 97 | 93 | 98 |
| `mx.fast.rms_norm`, GB/s | 74 | 107 | 109 | 109 |

So this is not the memory system, not the reduction width, and not the
hardware. The weight and the bias are not the cause either: layer_norm with
no weight and no bias still gives 36 GB/s at `D = 128`.

Twelve of the fourteen appendix shapes use `d_model` 128 or 32, so they all
took the slow path. `fast_layernorm.py` now replaces the kernel below a width
of 256, and it gave **1.205x** on the FLOP-weighted sweep. See
`OPTIMIZATIONS.md` row 31, which is KEPT.

## An MLX transpose is a free view

`x.transpose(0, 2, 1, 3)` costs nothing. It builds a strided view, and
`mx.eval` on it does not copy. Measured at B1024 S128 H4 hd32: the split
plus the transpose is 0.042 ms, and `mx.contiguous` of the same result is
3.55 ms.

The cost does not disappear. It moves into the kernel that consumes the
view, which then reads with a stride. See `OPTIMIZATIONS.md` row 32.

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

## The CPU and the GPU share one memory system

Unified memory is not only a convenience for zero-copy. It is a **shared
bandwidth budget**, and it decides which optimizations can work.

| Mover | Rate |
|---|---|
| GPU, measured roof | 128 GB/s |
| CPU, `mx.array(numpy)` copy | 46 GB/s |
| CPU, `np.array(mlx)` copy | 15 to 22 GB/s |

A CPU copy and a GPU kernel do NOT overlap. They contend. Measured on the
shape 6 chunk loop, which runs at 105% to 110% of the memory roof: an
unrelated 625 MiB CPU memcpy costs **31.40 ms alone and +45.06 ms inside the
loop**. Overlapping it is worse than running it on its own.

So do not carry a discrete-GPU instinct here. There a transfer crosses PCIe
while the GPU reads its own VRAM, and double buffering is free. On this
machine both sides use the one controller. See OPTIMIZATIONS.md row 48 and
`profiling/probes/pipeline_probe.py`.
