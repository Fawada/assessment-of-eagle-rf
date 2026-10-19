# Thread scaling: L0100N1504 z=0 (raw output of summarise_scaling.py, Slurm job 12186032)

n_estimators=1000, max_depth=32, n_train=48031, target bins=9, host=m8138.pri.cosma.local

| Threads | Runs | Fit min (s) | Fit median (s) | Fit max (s) | Speed-up | Efficiency | Peak RSS (GB) |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 120.92 | 121.04 | 121.18 | 1.00 | 1.00 | 8.23 |
| 2 | 3 | 60.99 | 61.09 | 61.30 | 1.98 | 0.99 | 8.43 |
| 4 | 3 | 30.82 | 30.92 | 35.90 | 3.91 | 0.98 | 8.48 |
| 8 | 3 | 15.71 | 16.16 | 16.87 | 7.49 | 0.94 | 8.52 |
| 16 | 3 | 8.61 | 8.66 | 8.87 | 13.98 | 0.87 | 8.75 |
| 32 | 3 | 4.97 | 5.10 | 5.23 | 23.73 | 0.74 | 8.80 |
| 64 | 3 | 2.88 | 3.01 | 3.07 | 40.23 | 0.63 | 8.86 |
| 256 | 3 | 2.65 | 2.67 | 2.72 | 45.41 | 0.18 | 9.47 |

Correctness: forest hash identical across all 24 runs: **yes** (5590010f3e7cacbb); distinct test R2 values (10 d.p.): [0.7419863541]

Note added after the run: the 256-thread row uses both SMT hardware threads of all 128 physical
cores (nproc reported 256). Its efficiency of 0.18 is relative to 256; relative to the 128 physical
cores it is 45.41 / 128 = 0.35. The 128-thread point (one thread per physical core) was not run in
this job; see job 12187354 / the pre-assessment report. `times.csv` in this folder excludes the
256-thread row.
