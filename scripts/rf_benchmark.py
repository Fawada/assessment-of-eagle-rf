#!/usr/bin/env python3
"""Benchmark harness for the EAGLE gas-density-profile Random Forest script.

The computation is the same as the owner's script: same HDF5 datasets, same cuts,
same feature/target transforms, same 80/20 split and the same RandomForestRegressor
hyperparameters and random_state.

Changes made for benchmarking (all optional, and listed in the results JSON):
  * paths and run parameters are command-line arguments, not hard-coded;
  * every phase is timed separately (read, preprocess, split, fit, predict, save, plot);
  * saving the model/arrays and plotting are OFF by default (switch on with flags);
  * n_jobs, n_estimators and an optional halo subsample are adjustable;
  * an extra prediction on the test set is timed, and the fitted forest is hashed, as a
    correctness check. With a fixed random_state the trees are bitwise identical for any
    n_jobs, so `forest_hash` must not change across a thread-scaling sweep. Predictions
    themselves differ at round-off level (~1e-15) between thread counts, because threads
    sum the per-tree outputs in a different order, so they are compared with a tolerance
    (`max_abs_pred`/`test_r2`), not hashed.

Usage (same three positional arguments as the original script):
    python rf_benchmark.py <Lbox> <Npart> <z> --outdir bench_out [--n-jobs 16] ...
Example:
    python rf_benchmark.py 0100 1504 0.0 --outdir bench_out --n-jobs 16
"""

from __future__ import annotations

import time

T_START = time.perf_counter()

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import socket
import sys
from pathlib import Path

import h5py
import numpy as np

RANDOM_STATE = 1  # as in the original script
# Data location: --data-root, or the EAGLE_DATA_ROOT environment variable. No default is built in,
# so the owner's directory is not published. Example: export EAGLE_DATA_ROOT=/path/to/eagle_data
DEFAULT_DATA_ROOT = os.environ.get("EAGLE_DATA_ROOT")

# Constants and radial binning, unchanged from the original script
LOGR_MIN = -2
LOGR_LOW = math.log10(0.05)
LOGR_MAX = math.log10(5.0)
LOGM_MIN = [10, 11.2, 12.3, 13.5]
LOGM_MAX = [11.2, 12.3, 13.5, 16]
H = 0.6777
G = 6.67e-11
MPC2KM = 3.0856776e19
KM2MPC = 1.0 / MPC2KM
M2KPC = KM2MPC
MSUN = 1.98847e30


def rho_crit_value() -> float:
    h0 = 100 * H * KM2MPC
    rho = 3 * h0**2 / (8 * np.pi * G)
    rho = rho / MSUN
    return rho * M2KPC**-3


def mod_arsinh(x, x0=1):
    return math.log10(math.exp(1)) * np.arcsinh(x / (2 * x0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("lbox", help="Box size string as in the file layout, e.g. 0100")
    p.add_argument("npart", help="Particle number string, e.g. 1504")
    p.add_argument("z", help="Redshift string as in the file name eagle_z<z>.hdf5")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="Directory containing L<lbox>N<npart>/ (default: $EAGLE_DATA_ROOT)")
    p.add_argument("--outdir", type=Path, default=Path("bench_out"))
    p.add_argument("--tag", default=None, help="Label for this run (default built from settings)")
    p.add_argument("--n-jobs", type=int, default=16, help="Threads for fit/predict (original: 16)")
    p.add_argument("--n-estimators", type=int, default=1000, help="Number of trees (original: 1000)")
    p.add_argument("--max-depth", type=int, default=32, help="Original: 32")
    p.add_argument("--min-samples-split", type=int, default=2, help="Original: 2")
    p.add_argument("--subsample", type=int, default=None,
                   help="Keep a random subset of this many haloes after the cuts (problem-size knob; off by default)")
    p.add_argument("--save-model", action="store_true", help="joblib.dump the model, as the original does")
    p.add_argument("--save-arrays", action="store_true", help="np.save the train/test arrays, as the original does")
    p.add_argument("--plot", action="store_true", help="Make the original 4-panel PDF figure")
    p.add_argument("--usetex", action="store_true", help="Use LaTeX for plot text, as the original does (needs latex on the node)")
    p.add_argument("--verbose", action="store_true", help="Print the arrays the original prints with test=True")
    return p.parse_args()


def forest_hash(model) -> str:
    """Hash of every tree's split features, thresholds, children and leaf values."""
    h = hashlib.sha256()
    for est in model.estimators_:
        t = est.tree_
        for arr in (t.feature, t.threshold, t.children_left, t.children_right, t.value):
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def library_versions() -> dict[str, str]:
    import joblib
    import sklearn
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass
    try:
        import matplotlib
        versions["matplotlib"] = matplotlib.__version__
    except ImportError:
        pass
    return versions


def make_plot(model, features, X_test, dens_prof_full, logr, r_mask, rho_crit, path, usetex):
    """Same figure as the original script."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rc("text", usetex=usetex)
    M200c = features[:, -1]
    M200c_test = X_test[:, -1]
    fig1 = plt.figure(figsize=(6 * len(LOGM_MIN), 6))
    fig1.subplots_adjust(hspace=0.0, wspace=0.0)
    fig1.suptitle(r"EAGLE-100", fontsize=28, y=1.05)
    for i in range(len(LOGM_MIN)):
        ax = plt.subplot(1, len(LOGM_MIN), i + 1)
        if i == len(LOGM_MIN) - 1:
            ax.set_title(r"$\log(M_{\rm 200c}/\rm M_{\odot}) > %s$" % (LOGM_MIN[i]), fontsize=25, y=1.01)
        else:
            ax.set_title(r"$%s < \log(M_{\rm 200c}/\rm M_{\odot}) < %s$" % (LOGM_MIN[i], LOGM_MAX[i]), fontsize=25, y=1.01)
        M_mask = np.logical_and(M200c < LOGM_MAX[i], M200c > LOGM_MIN[i])
        M_mask_test = np.logical_and(M200c_test < LOGM_MAX[i], M200c_test > LOGM_MIN[i])
        dens_plot_log = dens_prof_full[M_mask, :]
        features_plot = X_test[M_mask_test, :]
        if len(dens_plot_log) == 0 or len(features_plot) == 0:
            continue  # guard for small subsamples; the original assumes every bin is populated
        dens_plot_pred_log = model.predict(features_plot)
        dens_plot = 2 * rho_crit * np.sinh(dens_plot_log / math.log10(math.exp(1)), dtype=np.float64)
        dens_plot_pred = 2 * rho_crit * np.sinh(dens_plot_pred_log / math.log10(math.exp(1)), dtype=np.float64)
        dens_plot_mean = np.mean(dens_plot, axis=0, dtype=np.float64)
        dens_plot_std = np.std(dens_plot, axis=0, dtype=np.float64)
        dens_plot_pred_mean = np.mean(dens_plot_pred, axis=0, dtype=np.float64)
        dens_plot_pred_std = np.std(dens_plot_pred, axis=0, dtype=np.float64)
        ax.errorbar(10**logr, dens_plot_mean, yerr=dens_plot_std, capsize=3, color="sienna", label="True", zorder=2)
        ax.plot(10**logr[r_mask], dens_plot_pred_mean, c="sandybrown", lw=2, ls="--", label="Predicted", zorder=3)
        ax.fill_between(10**logr[r_mask], dens_plot_pred_mean - dens_plot_pred_std,
                        dens_plot_pred_mean + dens_plot_pred_std, fc="sandybrown", alpha=0.3, zorder=1)
        ax.annotate("N=%s" % (len(dens_plot)), size=20, xycoords="axes fraction", xy=(0.7, 0.9),
                    bbox=dict(boxstyle="round", fc="w"))
        ax.axvline(0.05, lw=1, c="black", ls=":")
        ax.axvline(1, lw=1, c="black", ls=":")
        ax.set_ylim(20, 1.0e8)
        ax.set_xlim(0.01, 5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$r/ R_{\rm 200c}$", fontsize=25)
        if i > 0:
            ax.tick_params(labelsize=22, labelleft=False, direction="in", which="both", pad=10)
        else:
            ax.tick_params(labelsize=22, direction="in", which="both", pad=10)
            ax.legend(loc="center right", fontsize=20, ncol=1)
            ax.set_ylabel(r"$\rho$ $\rm [M_{\odot} \, kpc^{-3}]$", fontsize=25)
    fig1.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig1)


def main() -> None:
    args = parse_args()
    if not args.data_root:
        raise SystemExit("No data location: pass --data-root /path/to/eagle_data or set EAGLE_DATA_ROOT")
    timings: dict[str, float] = {}
    mod = "L%sN%s" % (args.lbox, args.npart)
    tag = args.tag or f"{mod}_z{args.z}_nj{args.n_jobs}_ne{args.n_estimators}" + (
        f"_sub{args.subsample}" if args.subsample else "")
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rho_crit = rho_crit_value()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    timings["startup"] = time.perf_counter() - T_START

    # ---- Read (I/O) ----
    prof_file = Path(args.data_root) / mod / f"eagle_z{args.z}.hdf5"
    t = time.perf_counter()
    with h5py.File(prof_file, "r") as pf:
        dens_prof = np.asarray(pf["DensityProfiles/Gas"])
        Mgas_gal = np.asarray(pf["CentralGalaxyMass_Gas"])
        Mstar_gal = np.asarray(pf["CentralGalaxyMass_Stellar"])
        SFR_gal = np.asarray(pf["CentralGalaxyStarFormationRate"])
        Mbh_gal = np.asarray(pf["CentralGalaxyMass_BH"])
        Mbhdot = np.asarray(pf["CentralGalaxyBHAccretionRate"])
        Mtot_gal = np.asarray(pf["CentralGalaxyMass"])  # read but unused, as in the original
        M200c = np.asarray(pf["MCrit200"])
    timings["read"] = time.perf_counter() - t
    input_bytes = prof_file.stat().st_size

    # ---- Preprocess (serial numpy) ----
    t = time.perf_counter()
    features = np.vstack((Mgas_gal, Mstar_gal, SFR_gal, Mbh_gal, Mbhdot, M200c)).T
    Nprof, Nbins = np.shape(dens_prof)
    logr = np.linspace(LOGR_MIN, LOGR_MAX, Nbins)
    r_mask = np.logical_and(logr > LOGR_LOW, logr < 0)
    dens_prof_full = np.copy(dens_prof)
    dens_prof = dens_prof[:, r_mask]
    features = features[:Nprof, :]
    Mgas_gal = Mgas_gal[:Nprof]
    Mstar_gal = Mstar_gal[:Nprof]
    M200c = M200c[:Nprof]
    mask_final = np.logical_and(np.logical_and(Mgas_gal > 0, Mstar_gal > 0), M200c > 1.0e10)
    features = features[mask_final, :]
    dens_prof_full = dens_prof_full[mask_final, :]
    dens_prof = dens_prof[mask_final]
    n_after_cuts = len(features)

    if args.subsample is not None:
        if args.subsample > n_after_cuts:
            raise SystemExit(f"--subsample {args.subsample} exceeds the {n_after_cuts} haloes after cuts")
        rng = np.random.default_rng(RANDOM_STATE)
        keep = np.sort(rng.choice(n_after_cuts, size=args.subsample, replace=False))
        features, dens_prof, dens_prof_full = features[keep], dens_prof[keep], dens_prof_full[keep]

    dens_prof_full = mod_arsinh(dens_prof_full, rho_crit)
    dens_prof = mod_arsinh(dens_prof, rho_crit)
    features[:, 0] = np.log10(features[:, 0])
    features[:, 1] = np.log10(features[:, 1])
    features[:, 2] = mod_arsinh(features[:, 2])
    features[:, 3] = mod_arsinh(features[:, 3])
    features[:, 4] = mod_arsinh(features[:, 4])
    features[:, 5] = np.log10(features[:, 5])
    timings["preprocess"] = time.perf_counter() - t

    if args.verbose:
        print("features shape:", features.shape)
        print(features[:10, :])
        print("density profiles shape:", dens_prof.shape)
        print(dens_prof[:10, :])

    # ---- Split ----
    t = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        features, dens_prof, train_size=0.8, random_state=RANDOM_STATE)
    timings["split"] = time.perf_counter() - t

    # ---- Fit (the benchmark kernel) ----
    model = RandomForestRegressor(
        min_samples_split=args.min_samples_split,
        max_depth=args.max_depth,
        max_features="sqrt",
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        random_state=RANDOM_STATE,
    )
    t = time.perf_counter()
    model.fit(X_train, y_train)
    timings["fit"] = time.perf_counter() - t

    # ---- Predict on the test set (added: correctness check) ----
    t = time.perf_counter()
    y_pred = model.predict(X_test)
    timings["predict"] = time.perf_counter() - t
    r2 = float(r2_score(y_test, y_pred))

    total_nodes = int(sum(e.tree_.node_count for e in model.estimators_))
    value_bytes = int(sum(e.tree_.value.nbytes for e in model.estimators_))
    max_tree_depth = int(max(e.tree_.max_depth for e in model.estimators_))

    # ---- Save (I/O, off by default) ----
    saved: dict[str, int] = {}
    t = time.perf_counter()
    if args.save_model:
        import joblib
        path = outdir / f"model_eagle_{mod}_z{args.z}_red.pkl"
        joblib.dump(model, path)
        saved[path.name] = path.stat().st_size
    if args.save_arrays:
        for name, arr in [("train_features", X_train), ("train_target", y_train),
                          ("test_features", X_test), ("test_target", y_test)]:
            path = outdir / f"{name}_eagle_{mod}_z{args.z}_red.npy"
            np.save(path, arr, allow_pickle=False)
            saved[path.name] = path.stat().st_size
    timings["save"] = time.perf_counter() - t

    # ---- Plot (off by default) ----
    t = time.perf_counter()
    if args.plot:
        path = outdir / f"RF_dens_prof_eagle_{mod}_z{args.z}_rev.pdf"
        make_plot(model, features, X_test, dens_prof_full, logr, r_mask, rho_crit, path, args.usetex)
        saved[path.name] = path.stat().st_size
    timings["plot"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - T_START
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    result = {
        "tag": tag,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpus_available": len(os.sched_getaffinity(0)),
        "thread_env": {k: os.environ.get(k) for k in
                       ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
        "versions": library_versions(),
        "settings": {
            "simulation": mod, "z": args.z, "n_jobs": args.n_jobs,
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split, "max_features": "sqrt",
            "random_state": RANDOM_STATE, "subsample": args.subsample,
            "save_model": args.save_model, "save_arrays": args.save_arrays,
            "plot": args.plot, "usetex": args.usetex,
        },
        "data": {
            "input_file": str(prof_file), "input_bytes": input_bytes,
            "haloes_in_file": int(Nprof), "radial_bins": int(Nbins),
            "haloes_after_cuts": int(n_after_cuts), "haloes_used": int(len(features)),
            "target_bins": int(dens_prof.shape[1]),
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        },
        "timings_s": {k: round(v, 4) for k, v in timings.items()},
        "memory": {"peak_rss_bytes": peak_rss_bytes},
        "model": {"total_nodes": total_nodes, "value_array_bytes": value_bytes,
                  "max_tree_depth_reached": max_tree_depth},
        "outputs_bytes": saved,
        "correctness": {
            "forest_hash": forest_hash(model),
            "test_r2_transformed_space": r2,
            "mean_pred": float(np.mean(y_pred)),
            "max_abs_pred": float(np.max(np.abs(y_pred))),
        },
    }
    out_json = outdir / f"result_{tag}.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[{tag}] fit={timings['fit']:.2f}s predict={timings['predict']:.2f}s "
          f"total={timings['total']:.2f}s peakRSS={peak_rss_bytes / 1e9:.2f}GB "
          f"r2={r2:.10f} forest={result['correctness']['forest_hash']} -> {out_json}")


if __name__ == "__main__":
    main()
