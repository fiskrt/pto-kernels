# Mega-kernel line counts

Comparison of original10748eb and current productionaac5a4d. Counts cover all
seven compiled source files in csrc/kernel/gdn_mega for each version, including
the entry point and its locally included implementation files. Excludes README,
tests, benchmarks, host wrappers and external PTO/CANN headers.

| Metric | Original | Current |
|---|---:|---:|
| Source files | 7 | 7 |
| Total physical lines | 5241 | 1302 |
| Nonblank lines (including comments) | 4737 | 1239 |
| Lines containing code (excluding blanks/comments) | 3500 | 1130 |

Total physical lines fell by75.2%. The current source often puts multiple statements on one line, so this does not imply an equal reduction in algorithmic complexity.

## Original

| File | Total | Nonblank | Code |
|---|---:|---:|---:|
| chunk_cumsum.cpp | 371 | 341 | 174 |
| chunk_h.cpp | 855 | 766 | 590 |
| chunk_o.cpp | 1237 | 1138 | 852 |
| kernel_gdn_mega.cpp | 589 | 511 | 461 |
| scaled_dot_kkt.cpp | 610 | 563 | 300 |
| tri_inverse_impl.cpp | 690 | 620 | 466 |
| wy_fast.cpp | 889 | 798 | 657 |

## Current

| File | Total | Nonblank | Code |
|---|---:|---:|---:|
| common.hpp | 155 | 138 | 105 |
| fused_pipeline.hpp | 557 | 541 | 506 |
| gate_scan.hpp | 111 | 110 | 106 |
| inverse.hpp | 54 | 51 | 46 |
| kernel_gdn_mega.cpp | 133 | 119 | 104 |
| recurrence_pipeline.hpp | 123 | 117 | 107 |
| shared_grams.hpp | 169 | 163 | 156 |

Method: split source into physical lines; count nonblank lines; then remove C/C++ line/block comments while preserving quoted strings and newlines and count remaining nonblank lines. Preprocessor directives and brace-only lines count as code.
