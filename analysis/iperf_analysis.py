from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

@dataclass
class IperfConfig:
    root_dir: Path  # e.g. /Users/kmcomer/Documents/5G Experiment Data
    output_dir: Path

    # filters
    locations: Optional[List[str]] = None      # ["normal","fair"]
    bandwidths: Optional[List[str]] = None     # ["20","40","80"] or ["20 MHz"]
    tdds: Optional[List[str]] = None           # ["7-2","5-4"]
    cids: Optional[List[str]] = None           # ["1","2",...]
    directions: Optional[List[str]] = None     # ["UL","DL"]

    # alignment
    nearest_tolerance_s: Optional[float] = 2.0  # set None for no tolerance

    # display
    show_plots: bool = True
    save_plots: bool = True


# ============================================================
# Helpers
# ============================================================

def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _norm_cid(cid: str) -> str:
    return str(int(str(cid)))


def _normalize_bw_token(s: str) -> str:
    s = str(s).strip()
    return s.replace(" MHz", "")


def _passes_filter(v: Any, allowed: Optional[List[str]]) -> bool:
    if allowed is None:
        return True
    return str(v) in {str(x) for x in allowed}


def _parse_config_folder(name: str) -> Optional[Tuple[str, str]]:
    # "20_7-2" -> ("20", "7-2")
    m = re.match(r"(?P<bw>\d+)[_\-](?P<tdd>\d-\d)$", name)
    if not m:
        return None
    return m.group("bw"), m.group("tdd")


def _to_datetime_utc(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col, utc=True, errors="coerce")


def _read_summary_csv(path: Path, direction: str, location: str, bandwidth_raw: str, tdd: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # summary has duplicate column name "ns"; pandas auto-renames second to ns.1
    # expected: start + ns + ... + end + ns.1
    if "start" in df.columns and "ns" in df.columns:
        df["start_ts"] = _to_datetime_utc(df["start"].astype(str) + "." + df["ns"].astype(str))
    else:
        df["start_ts"] = pd.NaT

    if "end" in df.columns and "ns.1" in df.columns:
        df["end_ts"] = _to_datetime_utc(df["end"].astype(str) + "." + df["ns.1"].astype(str))
    elif "end" in df.columns and "ns" in df.columns:
        # fallback if parser didn't create ns.1
        df["end_ts"] = _to_datetime_utc(df["end"].astype(str) + "." + df["ns"].astype(str))
    else:
        df["end_ts"] = pd.NaT

    # parse numeric columns from strings like "144MBytes"
    if "transfer(MB)" in df.columns:
        df["transfer_mb"] = pd.to_numeric(df["transfer(MB)"].astype(str).str.replace("MBytes", "", regex=False), errors="coerce")
    if "bandwidth(Mbps)" in df.columns:
        df["bandwidth_mbps"] = pd.to_numeric(df["bandwidth(Mbps)"], errors="coerce")

    df["direction"] = direction.upper()
    df["location"] = location
    df["bandwidth_raw"] = str(bandwidth_raw)
    df["bandwidth"] = f"{bandwidth_raw} MHz"
    df["tdd"] = tdd
    if "device" in df.columns:
        df["cid"] = df["device"].astype(str).str.extract(r"pi0?(\d+)")[0].fillna("").map(lambda x: _norm_cid(x) if x else x)
    else:
        df["cid"] = None
    return df


# ============================================================
# 1) Load summary CSVs across all configurations
# ============================================================

def load_iperf_summaries(cfg: IperfConfig) -> pd.DataFrame:
    rows = []
    location_roots = [
        ("normal", cfg.root_dir / "iperf_normal_locations"),
        ("fair", cfg.root_dir / "iperf_fair"),
    ]

    for location, base in location_roots:
        if not base.exists():
            continue
        if not _passes_filter(location, cfg.locations):
            continue

        for config_dir in base.iterdir():
            if not config_dir.is_dir():
                continue
            parsed = _parse_config_folder(config_dir.name)
            if not parsed:
                continue
            bw_raw, tdd = parsed

            if cfg.bandwidths is not None:
                allowed_bw = {_normalize_bw_token(x) for x in cfg.bandwidths}
                if bw_raw not in allowed_bw:
                    continue
            if not _passes_filter(tdd, cfg.tdds):
                continue

            for direction in ("UL", "DL"):
                fp = config_dir / f"{direction}.csv"
                if not fp.exists():
                    continue
                sdf = _read_summary_csv(fp, direction, location, bw_raw, tdd)
                if cfg.cids is not None:
                    allowed_cids = {_norm_cid(x) for x in cfg.cids}
                    sdf = sdf[sdf["cid"].isin(allowed_cids)]
                if cfg.directions is not None:
                    allowed_dirs = {d.upper() for d in cfg.directions}
                    sdf = sdf[sdf["direction"].isin(allowed_dirs)]
                if not sdf.empty:
                    rows.append(sdf)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================
# 2) Parse per-device txt reports (full iperf output)
# ============================================================

_IPERF_LINE_RE = re.compile(
    r"""
    ^\[\s*\d+\]\s+
    (?P<t0>\d+\.\d+)-(?P<t1>\d+\.\d+)\s+sec\s+
    (?P<transfer>[\d\.]+)\s+MBytes\s+
    (?P<bandwidth>[\d\.]+)\s+Mbits/sec
    (?:\s+(?P<rest>.*))?
    $
    """,
    re.VERBOSE,
)

def parse_iperf_txt(txt_path: Path, location: str, bw_raw: str, tdd: str) -> pd.DataFrame:
    """
    Parses interval rows from iperf txt report into a DataFrame.
    Also extracts trial start wall-clock from the 'connected ... on YYYY-MM-DD HH:MM:SS' line.
    """
    txt = txt_path.read_text(errors="ignore").splitlines()

    # metadata from filename: pi05_UL.txt
    m = re.match(r"pi0?(\d+)_((UL)|(DL))\.txt$", txt_path.name, re.IGNORECASE)
    cid = _norm_cid(m.group(1)) if m else None
    direction = m.group(2).upper() if m else None

    trial_start_wall = None
    for line in txt:
        # ... on 2026-02-24 23:17:44 (CET)
        mm = re.search(r"\bon\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\b", line)
        if mm:
            trial_start_wall = pd.to_datetime(mm.group(1), utc=False, errors="coerce")
            break

    rows = []
    for line in txt:
        line = line.strip()
        mm = _IPERF_LINE_RE.match(line)
        if not mm:
            continue

        t0 = float(mm.group("t0"))
        t1 = float(mm.group("t1"))
        # skip summary line like 0.0000-30.x sec if desired? keep it but mark
        is_total = (abs(t0 - 0.0) < 1e-9) and (t1 > 5.0)

        rows.append({
            "source_file": str(txt_path),
            "cid": cid,
            "direction": direction,
            "interval_start_s": t0,
            "interval_end_s": t1,
            "interval_mid_s": (t0 + t1) / 2.0,
            "transfer_mb": float(mm.group("transfer")),
            "bandwidth_mbps": float(mm.group("bandwidth")),
            "is_total_line": is_total,
            "location": location,
            "bandwidth_raw": str(bw_raw),
            "bandwidth": f"{bw_raw} MHz",
            "tdd": tdd,
            "trial_start_wallclock": trial_start_wall,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Build wallclock interval midpoint if start timestamp available
    if pd.notna(df["trial_start_wallclock"]).any():
        # Note: txt time includes local timezone string (e.g., CET) not parsed robustly;
        # this stays naive/UTC-assumed unless you explicitly convert from CET.
        base = pd.to_datetime(df["trial_start_wallclock"], errors="coerce", utc=True)
        df["interval_mid_ts"] = base + pd.to_timedelta(df["interval_mid_s"], unit="s")
    else:
        df["interval_mid_ts"] = pd.NaT

    return df


def load_all_iperf_txt(cfg: IperfConfig) -> pd.DataFrame:
    rows = []
    location_roots = [
        ("normal", cfg.root_dir / "iperf_normal_locations"),
        ("fair", cfg.root_dir / "iperf_fair"),
    ]

    for location, base in location_roots:
        if not base.exists():
            continue
        if not _passes_filter(location, cfg.locations):
            continue

        for config_dir in base.iterdir():
            if not config_dir.is_dir():
                continue
            parsed = _parse_config_folder(config_dir.name)
            if not parsed:
                continue
            bw_raw, tdd = parsed

            if cfg.bandwidths is not None:
                allowed_bw = {_normalize_bw_token(x) for x in cfg.bandwidths}
                if bw_raw not in allowed_bw:
                    continue
            if not _passes_filter(tdd, cfg.tdds):
                continue

            for txt_path in config_dir.glob("pi*_*.txt"):
                df = parse_iperf_txt(txt_path, location, bw_raw, tdd)
                if df.empty:
                    continue

                if cfg.cids is not None:
                    allowed_cids = {_norm_cid(x) for x in cfg.cids}
                    df = df[df["cid"].isin(allowed_cids)]
                if cfg.directions is not None:
                    allowed_dirs = {d.upper() for d in cfg.directions}
                    df = df[df["direction"].isin(allowed_dirs)]

                if not df.empty:
                    rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ============================================================
# 3) Load physical layer ue_*.csv and align nearest timestamp
# ============================================================

def load_phys_layer_for_iperf(cfg: IperfConfig, phys_metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Reads files from folders like pi05_UL/ue_708a.csv and returns long-format rows.
    """
    rows = []
    location_roots = [
        ("normal", cfg.root_dir / "iperf_normal_locations"),
        ("fair", cfg.root_dir / "iperf_fair"),
    ]

    for location, base in location_roots:
        if not base.exists():
            continue
        if not _passes_filter(location, cfg.locations):
            continue

        for config_dir in base.iterdir():
            if not config_dir.is_dir():
                continue
            parsed = _parse_config_folder(config_dir.name)
            if not parsed:
                continue
            bw_raw, tdd = parsed

            if cfg.bandwidths is not None:
                allowed_bw = {_normalize_bw_token(x) for x in cfg.bandwidths}
                if bw_raw not in allowed_bw:
                    continue
            if not _passes_filter(tdd, cfg.tdds):
                continue

            for folder in config_dir.iterdir():
                if not folder.is_dir():
                    continue
                m = re.match(r"pi0?(\d+)_((UL)|(DL)).csv$", folder.name, re.IGNORECASE)
                if not m:
                    continue
                cid = _norm_cid(m.group(1))
                direction = m.group(2).upper()

                if cfg.cids is not None and cid not in {_norm_cid(x) for x in cfg.cids}:
                    continue
                if cfg.directions is not None and direction not in {d.upper() for d in cfg.directions}:
                    continue

                for ue_fp in folder.glob("ue_*.csv"):
                    try:
                        pldf = pl.read_csv(str(ue_fp), try_parse_dates=True)
                    except Exception:
                        continue

                    if "timestamp" not in pldf.columns:
                        continue
                    if pldf.schema["timestamp"] == pl.Utf8:
                        pldf = pldf.with_columns(pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False))
                    keep = ["timestamp"]
                    if phys_metrics:
                        keep += [c for c in phys_metrics if c in pldf.columns]
                    else:
                        # default common subset
                        for c in ["rssi", "rsrp", "rsrq", "sinr", "ulMcs", "dlMcs", "ulBytes", "dlBytes"]:
                            if c in pldf.columns:
                                keep.append(c)
                    keep = list(dict.fromkeys(keep))

                    pdf = pldf.select(keep).to_pandas()
                    pdf["location"] = location
                    pdf["bandwidth_raw"] = str(bw_raw)
                    pdf["bandwidth"] = f"{bw_raw} MHz"
                    pdf["tdd"] = tdd
                    pdf["cid"] = cid
                    pdf["direction"] = direction
                    pdf["rnti"] = ue_fp.stem.replace("ue_", "")
                    rows.append(pdf)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def align_iperf_with_phys(
    iperf_df: pd.DataFrame,
    phys_df: pd.DataFrame,
    tolerance_s: Optional[float] = 2.0,
) -> pd.DataFrame:
    """
    Nearest-time alignment using merge_asof on:
      location, bandwidth_raw, tdd, cid, direction
    iperf timestamp used:
      - interval_mid_ts if available
      - else start_ts (for summary rows)
    """
    if iperf_df.empty or phys_df.empty:
        return pd.DataFrame()

    left = iperf_df.copy()
    if "interval_mid_ts" in left.columns:
        left["align_ts"] = pd.to_datetime(left["interval_mid_ts"], utc=True, errors="coerce")
    elif "start_ts" in left.columns:
        left["align_ts"] = pd.to_datetime(left["start_ts"], utc=True, errors="coerce")
    else:
        return pd.DataFrame()

    right = phys_df.copy()
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")

    join_keys = ["location", "bandwidth_raw", "tdd", "cid", "direction"]
    for k in join_keys:
        if k not in left.columns or k not in right.columns:
            return pd.DataFrame()

    # merge_asof requires the merge keys to have identical datetime dtype/precision.
    # Explicitly align both to timezone-aware ns precision to avoid ns/us mismatch.
    left["align_ts"] = left["align_ts"].astype("datetime64[ns, UTC]")
    right["timestamp"] = right["timestamp"].astype("datetime64[ns, UTC]")

    left = left.sort_values("align_ts")
    right = right.sort_values("timestamp")

    merged = pd.merge_asof(
        left,
        right,
        left_on="align_ts",
        right_on="timestamp",
        by=join_keys,
        direction="nearest",
        tolerance=(pd.Timedelta(seconds=tolerance_s) if tolerance_s is not None else None),
    )

    if "timestamp" in merged.columns:
        merged["dt_abs_s"] = (merged["align_ts"] - merged["timestamp"]).abs().dt.total_seconds()
    return merged


# ============================================================
# 4) Plotting summary-level aggregates + per-device
# ============================================================

def plot_summary_aggregates(summary_df: pd.DataFrame, cfg: IperfConfig):
    if summary_df.empty:
        return

    # aggregate by config/location/direction
    agg = (
        summary_df
        .groupby(["location", "bandwidth", "tdd", "direction"], dropna=False)["bandwidth_mbps"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["config"] = agg["bandwidth"] + " | " + agg["tdd"] + " | " + agg["location"]

    plt.figure(figsize=(12, 5))
    sns.barplot(data=agg, x="config", y="mean", hue="direction", errorbar=None)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Mean iperf bandwidth (Mbps)")
    plt.title("Iperf aggregate throughput by configuration")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if cfg.save_plots:
        plt.savefig(cfg.output_dir / "iperf_aggregate_by_config.pdf")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close()


def plot_summary_by_device(summary_df: pd.DataFrame, cfg: IperfConfig):
    if summary_df.empty:
        return

    summary_df = summary_df.copy()
    summary_df["config"] = summary_df["bandwidth"] + " | " + summary_df["tdd"] + " | " + summary_df["location"]

    g = sns.catplot(
        data=summary_df,
        x="cid",
        y="bandwidth_mbps",
        hue="direction",
        col="config",
        kind="violin",
        split=True,
        inner="quart",
        cut=0,
        col_wrap=3,
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Device CID", "Iperf bandwidth (Mbps)")
    g.set_titles("{col_name}")
    for ax in g.axes.flatten():
        ax.grid(True, axis="y", alpha=0.3)
    g.figure.tight_layout()
    if cfg.save_plots:
        g.savefig(cfg.output_dir / "iperf_by_device_per_config.pdf")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(g.figure)


# ============================================================
# 5) End-to-end runner
# ============================================================

def run_iperf_analysis(cfg: IperfConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_iperf_summaries(cfg)
    txt_df = load_all_iperf_txt(cfg)
    phys_df = load_phys_layer_for_iperf(cfg)
    if phys_df.empty:
        print('empty')
    aligned_txt_phys = align_iperf_with_phys(txt_df, phys_df, tolerance_s=cfg.nearest_tolerance_s)
    aligned_summary_phys = align_iperf_with_phys(summary_df, phys_df, tolerance_s=cfg.nearest_tolerance_s)

    # save tables
    if not summary_df.empty:
        summary_df.to_csv(cfg.output_dir / "iperf_summary_rows.csv", index=False)
    if not txt_df.empty:
        txt_df.to_csv(cfg.output_dir / "iperf_txt_intervals.csv", index=False)
    if not aligned_txt_phys.empty:
        aligned_txt_phys.to_csv(cfg.output_dir / "iperf_txt_aligned_with_phys.csv", index=False)
    if not aligned_summary_phys.empty:
        aligned_summary_phys.to_csv(cfg.output_dir / "iperf_summary_aligned_with_phys.csv", index=False)

    # plots
    plot_summary_aggregates(summary_df, cfg)
    plot_summary_by_device(summary_df, cfg)

    print(f"summary rows: {len(summary_df)}")
    print(f"txt interval rows: {len(txt_df)}")
    print(f"aligned txt+phys rows: {len(aligned_txt_phys)}")
    print(f"aligned summary+phys rows: {len(aligned_summary_phys)}")


if __name__ == "__main__":
    cfg = IperfConfig(
        root_dir=Path("/Users/kmcomer/Documents/5G Experiment Data"),
        output_dir=Path.cwd() / "iperf_analysis_outputs",
        locations=["normal", "fair"],
        bandwidths=["20", "40", "80"],
        tdds=None,
        cids=["1", "2", "3", "4", "5", "6"],
        directions=["UL", "DL"],
        nearest_tolerance_s=2.0,
        show_plots=True,
        save_plots=True,
    )
    run_iperf_analysis(cfg)