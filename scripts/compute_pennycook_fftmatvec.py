#!/usr/bin/env python3
"""Compute Pennycook PP for FFTMatvec hipcc vs SCALE-amd from -raw logs.

-raw layout (9 numeric rows):
  rows 0-2: summary min/mean/max for [Initialize, Setup, Matvecs]  (full-run; excluded)
  rows 3-5: F  matvec per-iter min/mean/max over phases
  rows 6-8: F* matvec per-iter min/mean/max over phases

Per-matvec phase columns:
  Broadcast, Pad, FFT, SOTI-to-TOSI, SBGEMV, TOSI-to-SOTI, IFFT, Unpad, Reduce, Total
"""

import sys
from pathlib import Path

PHASES = [
    "Broadcast",
    "Pad",
    "FFT",
    "SOTI-to-TOSI",
    "SBGEMV",
    "TOSI-to-SOTI",
    "IFFT",
    "Unpad",
    "Reduce",
    "Total",
]

METRICS = {
    # Sum of GPU/compute phases only (excludes MPI Broadcast/Reduce and the Total column).
    "pipeline": list(range(1, 8)),
    # Core matrix-vector multiply kernel only.
    "sbgemv": [4],
    # Per-matvec Total column from -raw (includes Broadcast/Reduce).
    "total": [9],
}


def parse_numeric_rows(path):
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln[0].isalpha():
            continue
        try:
            rows.append([float(x) for x in ln.split()])
        except ValueError:
            pass
    if len(rows) < 9:
        raise SystemExit(f"Need >=9 numeric rows in {path}, got {len(rows)}")
    return rows[-9:]


def phase_time(row, cols):
    return sum(row[c] for c in cols)


def parse_raw_log(path, metric="pipeline"):
    if metric not in METRICS:
        raise SystemExit(f"Unknown metric {metric!r}; choose from {list(METRICS)}")
    cols = METRICS[metric]
    rows = parse_numeric_rows(path)
    # rows[0:3] = full-run summary (Initialize/Setup/Matvecs) — intentionally skipped
    f_mean = rows[4]
    fs_mean = rows[7]
    if len(f_mean) < 10 or len(fs_mean) < 10:
        raise SystemExit(f"Expected 10 phase columns in {path}, got {len(f_mean)}")
    return phase_time(f_mean, cols), phase_time(fs_mean, cols)


def pp(t_native, t_scale):
    tmin = min(t_native, t_scale)
    return 0.5 * (tmin / t_native + tmin / t_scale)


def main():
    hip_log = Path(sys.argv[1])
    scale_log = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("pennycook_fftmatvec.csv")
    metric = sys.argv[4] if len(sys.argv) > 4 else "pipeline"

    f_hip, fs_hip = parse_raw_log(hip_log, metric)
    f_sc, fs_sc = parse_raw_log(scale_log, metric)

    pp_f = pp(f_hip, f_sc)
    pp_fs = pp(fs_hip, fs_sc)
    suite = 0.5 * (pp_f + pp_fs)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write(f"metric,{metric}\n")
        f.write("op,hipcc_s,scale_amd_s,ratio_scale_over_hip,pp,faster\n")
        for op, th, ts in [("F", f_hip, f_sc), ("Fstar", fs_hip, fs_sc)]:
            faster = "scale-amd" if ts < th else "hipcc"
            f.write(f"{op},{th:.9f},{ts:.9f},{ts/th:.6f},{pp(th, ts):.6f},{faster}\n")
        f.write(f"SUITE,,,,{suite:.6f},\n")

    print(f"Metric: {metric} (raw per-matvec mean rows; excludes Initialize/Setup/Matvecs summary)")
    print(f"F  matvec: hipcc={f_hip:.6f}s  scale={f_sc:.6f}s  PP={pp_f:.4f}")
    print(f"F* matvec: hipcc={fs_hip:.6f}s  scale={fs_sc:.6f}s  PP={pp_fs:.4f}")
    print(f"Suite PP: {suite:.4f}")
    print(f"CSV: {out}")


if __name__ == "__main__":
    main()
