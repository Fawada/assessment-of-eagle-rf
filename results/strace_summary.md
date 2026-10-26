# strace I/O cross-check (Lustre, owner's full configuration)

Raw output of summarise_strace.py for the 3-repeat strace job (run_strace.sh), 2026-10-21.

| Repeat | Plain: save / total (s) | strace: save / total (s) | t_read (s) | t_write (s) | write calls | Write speed | C_I/O (strace) |
|---|---|---|---|---|---|---|---|
| 1 | 19.86 / 33.13 | 31.20 / 46.93 | 0.07 | 30.07 | 2048 | 243 MB/s | 0.358 |
| 2 | 15.96 / 28.71 | 23.42 / 38.76 | 0.07 | 22.28 | 2048 | 328 MB/s | 0.423 |
| 3 | 30.65 / 43.01 | 20.62 / 35.99 | 0.08 | 19.37 | 2048 | 377 MB/s | 0.460 |

| Summary over repeats | Value |
|---|---|
| Plain save time: median (min–max) | 19.86 s (15.96–30.65) |
| strace save time: median (min–max) | 23.42 s (20.62–31.20) |
| Plain total: median (min–max) | 33.13 s (28.71–43.01) |
| strace total: median (min–max) | 38.76 s (35.99–46.93) |
| Save time not spent in write() (pickling etc.), median | 1.14 s |
| open/close/fsync/lseek (not in the metric), median | 0.58 s |
| **C_I/O (strace): median (min–max)** | **0.423 (0.358–0.460)** |

Forest hash identical across all 6 runs: **yes**
Note: strace -c -w sums wall time over all threads; the save phase is single-threaded.

---
Earlier single-pair strace job (2026-10-21), one plain and one strace run:

| Run | Save (s) | Total (s) | t_read (s) | t_write (s) | C_I/O |
|---|---|---|---|---|---|
| plain | 37.07 | 54.77 | – | – | – |
| strace -f -c -w | 14.85 | 30.59 | 0.08 | 13.59 | 0.553 |

The apparent "-44% strace overhead" in that pair was Lustre's run-to-run variation, which is
why the repeated, alternating job above was run.
