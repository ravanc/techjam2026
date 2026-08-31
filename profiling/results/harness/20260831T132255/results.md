# Harness sweep

- script: `torch_transformer_benchmark.py` (unchanged)
- date: 2026-08-31T15:08:16
- commit: 311a420*
- device: cpu, dtype: float32
- torch: 2.13.0
- elapsed: 6320.9 s

`baseline` is `BaselineTransformer`. `optimized` is `UserOptimizedTransformer`. Both times are the median of the harness rounds.

| # | Shape (B, D, H, S, L) | Accuracy | baseline ms | optimized ms | speedup | optimized token/s |
|---:|---|:---:|---:|---:|---:|---:|
| 1 | 64, 128, 4, 128, 4 | PASS | 45.9754 | 3.2652 | 14.080x | 2,508,875 |
| 2 | 1, 128, 4, 128, 4 | PASS | 1.4456 | 0.6104 | 2.368x | 209,707 |
| 3 | 4, 128, 4, 128, 4 | PASS | 4.0834 | 0.6723 | 6.074x | 761,598 |
| 4 | 16, 128, 4, 128, 4 | PASS | 13.9391 | 1.2029 | 11.588x | 1,702,617 |
| 5 | 128, 128, 4, 128, 4 | PASS | 102.2755 | 5.8422 | 17.506x | 2,804,429 |
| 6 | 10000, 128, 4, 128, 4 | PASS | 15419.7160 | 436.8107 | 35.301x | 2,930,331 |
| 7 | 64, 32, 4, 128, 4 | PASS | 23.6896 | 1.0029 | 23.622x | 8,168,516 |
| 8 | 64, 1024, 4, 128, 4 | PASS | 461.1526 | 115.5255 | 3.992x | 70,911 |
| 9 | 64, 128, 1, 128, 4 | PASS | 26.3907 | 3.3803 | 7.807x | 2,423,475 |
| 10 | 64, 128, 2, 128, 4 | PASS | 37.3346 | 3.2754 | 11.399x | 2,501,088 |
| 11 | 64, 128, 16, 128, 4 | PASS | 121.8538 | 3.2514 | 37.478x | 2,519,549 |
| 12 | 64, 128, 4, 32, 4 | PASS | 7.5305 | 1.1551 | 6.519x | 1,773,031 |
| 13 | 64, 128, 4, 1024, 4 | PASS | 1845.8785 | 37.4570 | 49.280x | 1,749,632 |

## Summary

- cases run: 13
- cases that pass accuracy: 13
- median speedup: 11.588x
- minimum speedup: 2.368x (shape 2)
- maximum speedup: 49.280x (shape 13)
