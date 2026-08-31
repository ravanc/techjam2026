"""Measure where a copy happens between torch, numpy and MLX.

Apple Silicon puts the CPU and the GPU on one DRAM. That removes the bus.
It does not remove the copy. A copy happens when two allocators own two
buffers. This script shows which direction copies, and what the copy costs.

    .venv/bin/python3 profiling/probes/zero_copy.py
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

MIB = 2 ** 20
BYTES = 64 * MIB
COUNT = BYTES // 4


def median_ms(function, repeats=20):
    """Return the median wall time of one call, in milliseconds."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return sorted(samples)[len(samples) // 2] / 1e6


def report(name, function, nbytes=BYTES):
    ms = median_ms(function)
    print(f"{name:34s} {ms:8.4f} ms  {nbytes / ms / 1e6:9.1f} GB/s")


def alias_tests():
    """Show which direction gives a view, and which gives a copy."""
    print("== does the destination alias the source? ==")

    source = np.arange(8, dtype=np.float32)
    array = mx.array(source)
    mx.eval(array)
    source[0] = 99.0
    print(f"numpy -> mlx  mx.array(n)      alias={np.array(array)[0] == 99.0}")

    array = mx.arange(8, dtype=mx.float32)
    mx.eval(array)
    view = np.asarray(array)
    view[0] = 77.0
    mx.eval(array)
    print(f"mlx -> numpy  np.asarray(a)    alias={float(array[0]) == 77.0}"
          f"  owndata={view.flags['OWNDATA']}")

    copy = np.array(array)
    copy[0] = 11.0
    print(f"mlx -> numpy  np.array(a)      alias={float(array[0]) == 11.0}")

    host = torch.arange(8, dtype=torch.float32)
    device = host.to("mps")
    torch.mps.synchronize()
    host[0] = 55.0
    torch.mps.synchronize()
    print(f"torch cpu -> mps  .to('mps')   alias={float(device.cpu()[0]) == 55.0}"
          f"  cpu_ptr={hex(host.data_ptr())} mps_ptr={hex(device.data_ptr())}")

    try:
        device.numpy()
        print("torch mps -> numpy             ok")
    except TypeError as error:
        print(f"torch mps -> numpy             raises {str(error)[:52]}...")


def mps_import_race():
    """MLX reads an MPS buffer without waiting for the Metal queue."""
    print("\n== mx.array(torch mps tensor): a race ==")
    want = np.arange(1024, dtype=np.float32) + 1.0
    for label, sync in (("without synchronize", False), ("with synchronize", True)):
        wrong = 0
        for _ in range(10):
            tensor = torch.arange(1024, dtype=torch.float32, device="mps") + 1.0
            if sync:
                torch.mps.synchronize()
            array = mx.array(tensor)
            mx.eval(array)
            if not np.array_equal(np.array(array), want):
                wrong += 1
        print(f"{label:22s} wrong results: {wrong}/10")


def copy_rates():
    print(f"\n== copy rate, {BYTES / MIB:.0f} MiB float32 ==")
    source = np.random.rand(COUNT).astype(np.float32)
    host = torch.from_numpy(source)
    destination_np = np.empty_like(source)
    destination_mps = torch.empty(COUNT, dtype=torch.float32, device="mps")
    array = mx.array(source)
    mx.eval(array)
    mx.synchronize()

    def numpy_into_preallocated():
        destination_np[:] = source

    def mps_into_preallocated():
        destination_mps.copy_(host)
        torch.mps.synchronize()

    def torch_to_mps():
        host.to("mps")
        torch.mps.synchronize()

    def numpy_to_mlx():
        new = mx.array(source)
        mx.eval(new)
        mx.synchronize()

    def mlx_to_numpy_view():
        np.asarray(array)

    def mlx_to_numpy_copy():
        np.array(array)

    report("numpy -> numpy, preallocated", numpy_into_preallocated)
    report("torch cpu -> mps, preallocated", mps_into_preallocated)
    report("torch cpu -> mps, .to('mps')", torch_to_mps)
    report("numpy -> mlx, mx.array", numpy_to_mlx)
    report("mlx -> numpy, np.array (copy)", mlx_to_numpy_copy)
    report("mlx -> numpy, np.asarray (view)", mlx_to_numpy_view)


def one_array_two_streams():
    """A device is a property of the operation, not of the array."""
    print("\n== one array, both streams, no transfer call ==")
    array = mx.random.normal((2048, 2048))
    mx.eval(array)
    mx.synchronize()
    before = np.asarray(array).__array_interface__["data"][0]
    on_gpu = mx.sum(array * 2, stream=mx.gpu)
    on_cpu = mx.sum(array * 2, stream=mx.cpu)
    mx.eval(on_gpu, on_cpu)
    mx.synchronize()
    after = np.asarray(array).__array_interface__["data"][0]
    print(f"gpu={float(on_gpu):.3f} cpu={float(on_cpu):.3f} "
          f"same_buffer={before == after} ptr={hex(before)}")


def boundary_cost():
    """Cost of the conversion that `forward()` runs inside the timed region."""
    from torch_transformer_benchmark import _to_mlx, _to_torch

    print("\n== boundary cost at the Appendix 3.7 shapes ==")
    print(f"{'shape':22s} {'MiB':>8s} {'_to_mlx':>10s} "
          f"{'_to_torch':>11s} {'asarray view':>13s}")
    cases = (("1  B64 D128", (64, 128, 128)),
             ("8  B64 D1024", (64, 128, 1024)),
             ("6  B10000 D128", (10000, 128, 128)))
    for label, (batch, seq, width) in cases:
        x = torch.randn(batch, seq, width)
        array = _to_mlx(x)
        mx.eval(array)
        mx.synchronize()

        def to_mlx():
            new = _to_mlx(x)
            mx.eval(new)
            mx.synchronize()

        def to_torch():
            _to_torch(array, torch.float32, torch.device("cpu"))

        def to_torch_view():
            torch.from_numpy(np.asarray(array))

        megabytes = x.numel() * 4 / MIB
        print(f"{label:22s} {megabytes:8.1f} "
              f"{median_ms(to_mlx, 7):9.3f}m {median_ms(to_torch, 7):10.3f}m "
              f"{median_ms(to_torch_view, 7):12.4f}m")
        del x, array


def forward_breakdown():
    """
    Split `forward()` into the MLX part and the torch part.

    Every arithmetic operation runs in MLX. Four operations do not:
    two `_to_mlx` calls, one `valid_token_mask.all()` and one `_to_torch`.
    The timed region holds all four.
    """
    from torch_transformer_benchmark import (
        TransformerConfig, UserOptimizedTransformer, _to_mlx, _to_torch,
        generate_random_case,
    )

    cases = (
        ("2  B1 D128 S128", dict(batch_size=1, d_model=128, num_heads=4,
                                 seq_len=128, num_layers=4, ffn_dim=128,
                                 causal=True), 60),
        ("1  B64 D128 S128", dict(batch_size=64, d_model=128, num_heads=4,
                                  seq_len=128, num_layers=4, ffn_dim=128,
                                  causal=True), 40),
        ("8  B64 D1024 S128", dict(batch_size=64, d_model=1024, num_heads=4,
                                   seq_len=128, num_layers=4, ffn_dim=1024,
                                   causal=True), 15),
        ("6  B10000 D128 S128", dict(batch_size=10000, d_model=128,
                                     num_heads=4, seq_len=128, num_layers=4,
                                     ffn_dim=128, causal=True), 5),
    )

    print("\n== forward(): the MLX part and the torch part ==")
    print(f"{'shape':22s} {'total':>9s} {'to_mlx x':>9s} {'to_mlx m':>9s} "
          f"{'mask.all':>9s} {'mlx calc':>10s} {'to_torch':>9s} {'torch':>7s}")

    for label, keywords, repeats in cases:
        config = TransformerConfig(**keywords)
        model = UserOptimizedTransformer(config).eval()
        device = torch.device("cpu")
        x, valid = generate_random_case(config, device, torch.float32, 0, 0.0, 1.0)

        with torch.no_grad():
            for _ in range(3):
                model(x, valid)

            total = median_ms(lambda: model(x, valid), repeats)
            time_x = median_ms(
                lambda: (mx.eval(_to_mlx(x)), mx.synchronize()), repeats)
            time_mask = median_ms(
                lambda: (mx.eval(_to_mlx(valid)), mx.synchronize()), repeats)
            time_all = median_ms(lambda: bool(valid.all()), repeats)

            array_x, array_mask = _to_mlx(x), _to_mlx(valid)
            mx.eval(array_x, array_mask)
            mx.synchronize()
            call = model._mlx_call[False]
            chunk = model.plan.batch_chunk

            def calculate():
                if chunk is None:
                    out = call(array_x, array_mask, model._mlx_layers,
                               *model._mlx_final)
                    mx.eval(out)
                else:
                    parts = []
                    for start in range(0, array_x.shape[0], chunk):
                        stop = min(start + chunk, array_x.shape[0])
                        part = call(array_x[start:stop], array_mask[start:stop],
                                    model._mlx_layers, *model._mlx_final)
                        mx.eval(part)
                        parts.append(part)
                    mx.eval(mx.concatenate(parts, axis=0))
                mx.synchronize()

            time_calculate = median_ms(calculate, repeats)

            output = array_x if chunk is not None else call(
                array_x, array_mask, model._mlx_layers, *model._mlx_final)
            mx.eval(output)
            mx.synchronize()
            time_out = median_ms(
                lambda: _to_torch(output, torch.float32, device), repeats)

        share = 100 * (time_x + time_mask + time_all + time_out) / total
        print(f"{label:22s} {total:9.3f} {time_x:9.3f} {time_mask:9.3f} "
              f"{time_all:9.3f} {time_calculate:10.3f} {time_out:9.3f} "
              f"{share:6.1f}%")
        del model, x, valid, array_x, array_mask, output


if __name__ == "__main__":
    alias_tests()
    mps_import_race()
    copy_rates()
    one_array_two_streams()
    boundary_cost()
    forward_breakdown()
