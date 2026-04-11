from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import re

from data_loading import parse_experiment_name
from helpers import sort_experiments_by_sweep, format_sweep_label


THROUGHPUT_METRICS = {
    "ul_throughput_mbps",
    "dl_throughput_mbps",
    "ul_shannon",
    "dl_shannon",
    "ul_shannon_sinr",
    "dl_shannon_sinr",
    "ul_3gpp",
    "dl_3gpp"
}


# =========================
# Theoretical throughput
# =========================
data_size = 2.9172 * 8

mu = 1
T_s = 1e-3 / (14 *2 ** mu)
# R_max = 948 / 1024
f = 1
OH_dl = 0.14
OH_ul = 0.08
v_layers = 2

def calc_ratios(dl_slots, ul_slots, period_slots):
    flex_slots = period_slots - dl_slots - ul_slots
    # Flexible slot: 6/14 DL, 4/14 UL symbols
    eff_dl = (dl_slots + flex_slots * 6 / 14) / period_slots
    eff_ul = (ul_slots + flex_slots * 4 / 14) / period_slots
    return {'dl': eff_dl, 'ul': eff_ul}

# mu=1 -> 0.5ms slots -> 5ms = 10 slots, 2.5ms = 5 slots
slot_configs = {
    '7-2': calc_ratios(7, 2, 10),   # 1 flex slot
    '5-4': calc_ratios(5, 4, 10),   # 1 flex slot
    '2-7': calc_ratios(2, 7, 10),   # 1 flex slot
    '3-1': calc_ratios(3, 1, 5),    # 1 flex slot
    '2-2': calc_ratios(2, 2, 5),    # 1 flex slot
}
# N_PRB = {'20 MHz': 51, '40 MHz': 106, '60 MHz': 162,'80 MHz': 217, '100 MHz': 273}
N_PRB = {20: 51, 40: 106, 60: 162, 80: 217, 100: 273}
bandwidths = ['20 MHz', '40 MHz', '80 MHz', '100 MHz']

def theoretical_throughput(bw, Qm, R_max=948/1024, slot_config='7:2', direction='dl', num_users=1):
    oh = OH_dl if direction == 'dl' else OH_ul
    allocated_slots = slot_configs[slot_config][direction] / num_users
    return 1e-6 * v_layers * Qm * f * R_max * ((N_PRB[bw] * 12) / T_s) * (1 - oh) * allocated_slots

# =========================
# Config
# =========================

@dataclass
class PlotConfig:
    data_dir: Path
    output_dir: Path
    filters: Dict[str, Any]
    metrics: List[str]
    sweep: Optional[str] = None

    min_thresholds: Dict[str, float] = field(default_factory=dict)

    # pipeline behavior
    filter_rounds: bool = True
    annotate_phases: bool = True
    round_gap_s: int = 200

    # plotting behavior
    plot_mode: str = "distribution"          # "distribution" | "time"
    distribution_plot_type: str = "violin"   # violin|box|bar|count
    pair_ul_dl: bool = True
    show_plots: bool = True
    save_plots: bool = True
    smoothing: bool = False
    pts_to_plot: int = 1000
    pts_offset: int = 0

    # round profiles
    round_profiles_enabled: bool = False
    round_profile_points: int = 100000
    round_profile_phase_filter: Optional[List[str]] = None
    round_profile_round_ids: Optional[List[int]] = None
    round_profile_layout: str = "same_axes" # use 'subplots' for stacked UL/DL panels
    round_profile_error_bars: bool = False
    round_profile_errorbar_step: int = 10
    round_profile_include_effective_sum: bool = True
    round_profile_effective_secondary_axis: bool = True
    round_profile_time_mode: str = "normalized"   # "normalized" | "real_from_round_start"
    round_profile_curve_mode: str = "average"      # "average" | "per_round"
    round_profile_target_rnti: Optional[str] = None # plot this RNTI in per_round mode; default first available
    round_profile_target_rntis: Optional[List[str]] = None # per_round multi-RNTI selection; None uses all for multi modes
    round_profile_per_round_rnti_layout: str = "single"   # "single" | "same_axes" | "separate_figures"
    round_profile_max_curves: int = 12              # max number of round curves to overlay in per_round mode
    round_profile_ci_bands: bool = False
    round_profile_ci_level: float = 0.95
    round_profile_bin_s: float = 0.25   # for real-time mode binning

    # iperf
    dataset_type: str = "fedavg"   # "fedavg" | "iperf"
    iperf_root_dir: Optional[Path] = None
    cid_filter: Optional[List[str]] = None         # e.g. ["05", "06"] or ["5","6"]
    direction_filter: Optional[List[str]] = None   # ["UL","DL"]
    use_relative_time: bool = True


# =========================
# Small utilities
# =========================

def _safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')

def _norm_thresholds(d: Optional[Dict[str, float]]) -> Dict[str, float]:
    return {} if d is None else dict(d)


def _phase_list(phase_filter):
    if phase_filter is None:
        return None
    if isinstance(phase_filter, str):
        return None if phase_filter.lower() in {"all", "*"} else [phase_filter]
    out = list(phase_filter)
    return out or None


def _metric_values(df: pl.DataFrame, metric: str, thresholds: Dict[str, float]) -> List[float]:
    if metric not in df.columns:
        return []
    threshold = thresholds.get(metric)

    out = []
    for v in df[metric].to_list():
        if v is None or pd.isna(v):
            continue
        n = pd.to_numeric(v, errors="coerce")
        if pd.isna(n):
            continue
        if threshold is not None and n < threshold:
            continue
        out.append(float(n))
    return out


def _compute_mean_std(ue_dfs: Dict[str, pl.DataFrame], metric: str, thresholds: Dict[str, float]) -> Tuple[Optional[float], Optional[float]]:
    vals = []
    for df in ue_dfs.values():
        vals.extend(_metric_values(df, metric, thresholds))
    if not vals:
        return None, None
    s = pd.Series(vals)
    return float(s.mean()), (float(s.std()) if len(s) > 1 else None)

def _ci_halfwidth(std: np.ndarray, n: np.ndarray, level: float = 0.95) -> np.ndarray:
    # normal approx; switch to t-dist if you want exact small-sample behavior
    z = 1.96 if abs(level - 0.95) < 1e-9 else 1.96
    out = np.full_like(std, np.nan, dtype=float)
    mask = np.isfinite(std) & np.isfinite(n) & (n > 1)
    out[mask] = z * (std[mask] / np.sqrt(n[mask]))
    return out


# =========================
# Data loading
# =========================
def _metrics_to_load(metrics: List[str]) -> List[str]:
    base = [m for m in metrics if m not in THROUGHPUT_METRICS]
    if any(m in THROUGHPUT_METRICS for m in metrics):
        base += ["ulBytes", "dlBytes", "ulMcs", "dlMcs", "ulQm", "dlQm", "puschSnr", "sinr"]
    return list(dict.fromkeys(base))


def _read_joined_csv(main_fp: str, secondary_fp: Optional[str], columns: Optional[List[str]] = None) -> pl.DataFrame:
    if secondary_fp:
        left = pl.read_csv(main_fp, columns=["timestamp"], try_parse_dates=True).with_row_index("segment")
        right = pl.read_csv(secondary_fp, columns=(["segment"] + columns) if columns else None)
        return left.join(right, on="segment", how="right")
    return pl.read_csv(main_fp, columns=(["timestamp"] + columns) if columns else None, try_parse_dates=True)


def _add_throughput(df: pl.DataFrame, exp: dict) -> pl.DataFrame:
    # def theoretical_throughput(bw, Qm, R_max, slot_config='7:2', direction='dl', num_users=1):
    needed = {"timestamp", "ulBytes", "dlBytes", "ulQm", "dlQm", "ulMcs", "dlMcs", "puschSnr", "sinr"}
    if not needed.issubset(df.columns):
        return df
    
    ref_lut = (
        pl.read_csv(Path("ref.csv"), columns=["index", "mcs target code rate", "sinr"])
        .rename({"mcs target code rate": "mcs_target_code_rate", "sinr" : "sinr_map"})
        .with_columns(pl.col("index").cast(pl.Float64, strict=False))
        .with_columns(pl.col("sinr_map").cast(pl.Float64, strict=False))
    )
    bw = int(exp['bandwidth'].split(" ")[0])
    tdd = exp['tdd']
    dl_parts, ul_parts = map(int, tdd.split('-'))
    dl_pct = dl_parts / (dl_parts + ul_parts + 1)
    ul_pct = ul_parts / (dl_parts + ul_parts + 1)

#     ref_base = (
#     pl.read_csv(Path("ref.csv"), columns=["index", "mcs target code rate", "sinr"])
#     .rename({"mcs target code rate": "code_rate", "sinr": "sinr_map"})
#     .with_columns(pl.col("index").cast(pl.Float64, strict=False))
# )

# ul_lut = ref_base.select([
#     pl.col("mcs_index"),
#     pl.col("code_rate").alias("ul_code_rate"),
# ])

# dl_lut = ref_base.select([
#     pl.col("mcs_index"),
#     pl.col("code_rate").alias("dl_code_rate"),
# ])

# sinr_lut = ref_base.select([
#     pl.col("mcs_index"),
#     pl.col("sinr_map").alias("sinr_linear"),
# ])

# # Ensure left keys are same dtype
# df = df.with_columns(
#     pl.col("ulMcs").cast(pl.Float64, strict=False),
#     pl.col("dlMcs").cast(pl.Float64, strict=False),
#     pl.col("sinr").cast(pl.Float64, strict=False),
# )

# df = (
#     df.join(ul_lut, left_on="ulMcs", right_on="mcs_index", how="left")
#       .join(dl_lut, left_on="dlMcs", right_on="mcs_index", how="left")
#       .join(sinr_lut, left_on="sinr", right_on="mcs_index", how="left")
# )

    return (
        df.with_columns(
            ul_num=pl.col("ulBytes").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            dl_num=pl.col("dlBytes").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            ulMcs=pl.col("ulMcs").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            dlMcs=pl.col("dlMcs").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            ulQm=pl.col("ulQm").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            dlQm=pl.col("dlQm").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            sinr=pl.col("sinr").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            puschSnr=pl.col("puschSnr").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
        )
        .sort("timestamp")
        .with_columns(
            dt=pl.col("timestamp").diff().dt.total_seconds(),
            dul=pl.col("ul_num").diff(),
            ddl=pl.col("dl_num").diff(),
            puschSnr_linear=pl.lit(10.0).pow(pl.col("puschSnr") / 10.0),
            sinr_linear=pl.lit(10.0).pow(pl.col("sinr") / 10.0)
        )
        .join(
            ref_lut.rename({"mcs_target_code_rate": "ul_code_rate"}),
            left_on="ulMcs",
            right_on="index",
            how="left",
        )
        .join(
            ref_lut.rename({"mcs_target_code_rate": "dl_code_rate"}),
            left_on="dlMcs",
            right_on="index",
            how="left",
        )
        # .join(
        #     ref_lut.rename({"sinr_map": "sinr_linear"}),
        #     left_on="sinr",
        #     right_on="index",
        #     how="left",
        # )
        .with_columns(
            ul_throughput_mbps=pl.when((pl.col("dt") > 0) & (pl.col("dul") >= 0)).then((pl.col("dul") * 8 / 1_000_000) / pl.col("dt")).otherwise(None),
            dl_throughput_mbps=pl.when((pl.col("dt") > 0) & (pl.col("ddl") >= 0)).then((pl.col("ddl") * 8 / 1_000_000) / pl.col("dt")).otherwise(None),
            ul_shannon=ul_pct*bw*(1+pl.col("puschSnr_linear")).log(base=2),
            dl_shannon=dl_pct*bw*(1+pl.col("puschSnr_linear")).log(base=2),
            ul_shannon_sinr=ul_pct*bw*(1+pl.col("sinr_linear")).log(base=2),
            dl_shannon_sinr=dl_pct*bw*(1+pl.col("sinr_linear")).log(base=2),
            ul_3gpp = theoretical_throughput(bw, pl.col("ulQm"), pl.col("ul_code_rate"), tdd, "ul"),
            dl_3gpp = theoretical_throughput(bw, pl.col("dlQm"), pl.col("dl_code_rate"), tdd, "dl")
        )
        .drop("ul_num", "dl_num")
    )


def load_experiment_data(exp: dict, metrics: List[str]) -> Dict[str, pl.DataFrame]:
    exp_path = exp["path"]
    phys = exp_path / "phys_layer"
    if not phys.exists():
        return {}

    load_cols = _metrics_to_load(metrics)
    out = {}

    for fp in phys.iterdir():
        n = fp.name
        if "ue" not in n or "common" in n:
            continue
        rnti = n.split("_")[1].split(".")[0]
        df = _read_joined_csv(str(phys / "common.csv"), str(fp), load_cols)
        out[rnti] = _add_throughput(df, exp)

    return out

# iperf data loading
def build_iperf_experiment_index(root_dir: Path) -> List[dict]:
    exps = []
    location_dirs = [
        ("normal", root_dir / "iperf_normal_locations"),
        ("fair", root_dir / "iperf_fair"),
        ("all_connected", root_dir / "iperf_all_connected")
    ]

    for location, base in location_dirs:
        if not base.exists():
            continue
        for cfg_dir in base.iterdir():
            if not cfg_dir.is_dir():
                continue
            m = re.match(r"(?P<bw>\d+)[_\-](?P<tdd>\d-\d)$", cfg_dir.name)
            if not m:
                continue
            exps.append({
                "path": cfg_dir,
                "location": location,
                "bandwidth": f"{m.group('bw')} MHz",
                "bandwidth_raw": m.group("bw"),
                "tdd": m.group("tdd"),
            })
    return exps


def _normalize_cid(cid: str) -> str:
    # "05" -> "5", "5" -> "5"
    return str(int(str(cid)))


def load_iperf_experiment_data(
    exp_path: Path,
    metrics: List[str],
    cid_filter: Optional[List[str]] = None,
    direction_filter: Optional[List[str]] = None,
) -> Dict[str, pl.DataFrame]:
    cid_filter_norm = set(_normalize_cid(c) for c in cid_filter) if cid_filter else None
    direction_filter_norm = set(d.upper() for d in direction_filter) if direction_filter else None

    out = {}

    # folders like pi05_UL, pi06_DL
    for child in exp_path.iterdir():
        if not child.is_dir():
            continue
        
        m = re.match(r"pi0?(\d+)_((UL)|(DL)).csv$", child.name, re.IGNORECASE)
        if not m:
            continue
        cid_raw = m.group(1)           # e.g. "05" or "5"
        cid_norm = _normalize_cid(cid_raw)
        direction = m.group(2).upper()

        if cid_filter_norm and cid_norm not in cid_filter_norm:
            continue
        if direction_filter_norm and direction not in direction_filter_norm:
            continue
        for fp in child.glob("ue_*.csv"):
            df = pl.read_csv(str(fp), try_parse_dates=True)

            # ensure timestamp is datetime if read as string
            if "timestamp" in df.columns and df.schema["timestamp"] == pl.Utf8:
                df = df.with_columns(
                    pl.col("timestamp").str.to_datetime(time_zone="UTC", strict=False)
                )

            df = _add_throughput(df)

            rnti = fp.stem.replace("ue_", "")
            key = f"pi{int(cid_norm):02d}_{direction}_{rnti}"
            out[key] = df

    return out


# =========================
# Round filtering + phases
# (compact versions)
# =========================

def _get_duration_column(df: pd.DataFrame, candidates: List[str], default=0.0):
    for c in candidates:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").fillna(default)
            return np.maximum(vals, 0.0)
    return pd.Series(default, index=df.index, dtype="float64")


def build_round_windows(agg_metrics_file: str, max_gap_s=200) -> pd.DataFrame:
    agg = pd.read_csv(agg_metrics_file)
    agg["timestamp"] = pd.to_datetime(agg["timestamp"], unit="s", utc=True)
    agg = agg.sort_values("timestamp").reset_index(drop=True)

    gap_s = agg["timestamp"].diff().dt.total_seconds()
    agg["round_duration"] = _get_duration_column(agg, ["round_duration", "round_time", "duration_s"], default=np.nan)

    if agg["round_duration"].isna().all():
        inf = gap_s.copy()
        inf.iloc[0] = np.nan
        fallback = np.nanmedian(inf.to_numpy(dtype=float))
        fallback = 0.0 if np.isnan(fallback) else fallback
        agg["round_duration"] = inf.fillna(fallback)
    else:
        fallback = np.nanmedian(agg["round_duration"].to_numpy(dtype=float))
        fallback = 0.0 if np.isnan(fallback) else fallback
        agg["round_duration"] = pd.to_numeric(agg["round_duration"], errors="coerce").fillna(fallback)

    active = gap_s.isna() | (gap_s <= max_gap_s)
    r = agg.loc[active].copy().reset_index(drop=True)

    r["round_id"] = np.arange(len(r), dtype=int)
    r["round_end"] = r["timestamp"]
    r["round_start"] = r["round_end"] - pd.to_timedelta(r["round_duration"], unit="s")

    r["downlink_s"] = _get_duration_column(r, ["downlink_latency", "downlink_time", "dl_time_s"])
    r["train_s"] = _get_duration_column(r, ["train_time", "local_train_time", "training_time"])
    r["eval_s"] = _get_duration_column(r, ["eval_time", "evaluation_time"])
    r["uplink_s"] = _get_duration_column(r, ["uplink_latency", "uplink_time", "ul_time_s"])

    r["downlink_end"] = r["round_start"] + pd.to_timedelta(r["downlink_s"], unit="s")
    r["training_end"] = r["downlink_end"] + pd.to_timedelta(r["train_s"], unit="s")
    r["evaluation_end"] = r["training_end"] + pd.to_timedelta(r["eval_s"], unit="s")
    r["uplink_end"] = r["evaluation_end"] + pd.to_timedelta(r["uplink_s"], unit="s")
    return r


def annotate_telemetry_with_rounds_and_phases(trial_data: pl.DataFrame, rounds: pd.DataFrame) -> pl.DataFrame:
    if rounds.empty:
        return trial_data

    rp = pl.from_pandas(rounds[["round_id", "round_start", "round_end", "downlink_end", "training_end", "evaluation_end", "uplink_end"]])
    ts_dtype = trial_data.schema.get("timestamp")
    if ts_dtype is not None:
        rp = rp.with_columns(
            pl.col("round_start").cast(ts_dtype),
            pl.col("round_end").cast(ts_dtype),
            pl.col("downlink_end").cast(ts_dtype),
            pl.col("training_end").cast(ts_dtype),
            pl.col("evaluation_end").cast(ts_dtype),
            pl.col("uplink_end").cast(ts_dtype),
        )
    rp = rp.set_sorted("round_start")

    out = trial_data.join_asof(rp, left_on="timestamp", right_on="round_start", strategy="backward", check_sortedness=False)
    out = out.filter(pl.col("round_id").is_not_null() & (pl.col("timestamp") <= pl.col("round_end")))

    out = out.with_columns(
        phase=pl.when(pl.col("timestamp") <= pl.col("downlink_end")).then(pl.lit("downlink"))
        .when(pl.col("timestamp") <= pl.col("training_end")).then(pl.lit("training"))
        .when(pl.col("timestamp") <= pl.col("evaluation_end")).then(pl.lit("evaluation"))
        .when(pl.col("timestamp") <= pl.col("uplink_end")).then(pl.lit("uplink"))
        .otherwise(pl.lit("idle"))
    )
    return out


def apply_round_processing(exp_path: Path, ue_dfs: Dict[str, pl.DataFrame], cfg: PlotConfig):
    if not cfg.filter_rounds:
        return ue_dfs, pd.DataFrame()

    agg_fp = exp_path / "train_agg_metrics.csv"
    if not agg_fp.exists():
        return ue_dfs, pd.DataFrame()

    rounds = build_round_windows(str(agg_fp), max_gap_s=cfg.round_gap_s)
    if rounds.empty:
        return ue_dfs, rounds

    out = {}
    for rnti, df in ue_dfs.items():
        d = df.sort("timestamp")

        intervals = pl.from_pandas(rounds[["round_start", "round_end"]])
        ts_dtype = d.schema.get("timestamp")
        if ts_dtype is not None:
            intervals = intervals.with_columns(
                pl.col("round_start").cast(ts_dtype),
                pl.col("round_end").cast(ts_dtype),
            )
        intervals = intervals.set_sorted("round_start")

        d = d.join_asof(
            intervals,
            left_on="timestamp",
            right_on="round_start",
            strategy="backward",
            check_sortedness=False,
        )
        d = d.filter(
            pl.col("round_start").is_not_null() &
            (pl.col("timestamp") <= pl.col("round_end"))
        ).drop("round_start", "round_end")

        if cfg.annotate_phases:
            d = annotate_telemetry_with_rounds_and_phases(d, rounds)

        out[rnti] = d

    return out, rounds

# =========================
# Round profiles extension
# =========================
def compute_round_average_profile(
    annotated_trial_data: pl.DataFrame,
    metric: str,
    n_points: int = 100,
    phase_filter=None,
    round_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    if metric not in annotated_trial_data.columns:
        return pd.DataFrame()

    df = annotated_trial_data.to_pandas()
    phases = _phase_list(phase_filter)
    if phases is not None and "phase" in df.columns:
        df = df[df["phase"].isin(phases)]
    if round_ids is not None and "round_id" in df.columns:
        df = df[df["round_id"].isin(round_ids)]

    # round_t may not exist in compact pipeline; compute if missing
    if "round_t" not in df.columns:
        if not {"round_start", "round_end", "timestamp"}.issubset(df.columns):
            return pd.DataFrame()
        rs = pd.to_datetime(df["round_start"], utc=True)
        re = pd.to_datetime(df["round_end"], utc=True)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        dur = (re - rs).dt.total_seconds()
        el = (ts - rs).dt.total_seconds()
        df["round_t"] = np.where(dur > 0, el / dur, np.nan)

    df = df.dropna(subset=["round_id", "round_t", metric])
    if df.empty:
        return pd.DataFrame()

    grid = np.linspace(0.0, 1.0, n_points)
    aligned = []

    for rid, grp in df.groupby("round_id"):
        grp = grp.sort_values("round_t")
        x = grp["round_t"].to_numpy(dtype=float)
        y = pd.to_numeric(grp[metric], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue

        keep = np.r_[True, np.diff(x) > 0]
        x, y = x[keep], y[keep]
        if len(x) < 2:
            continue

        aligned.append(np.interp(grid, x, y))

    if not aligned:
        return pd.DataFrame()

    # arr = np.vstack(aligned)
    # return pd.DataFrame(
    #     {"round_t": grid, "mean": np.nanmean(arr, axis=0), "std": np.nanstd(arr, axis=0)}
    # )
    arr = np.vstack(aligned)
    return pd.DataFrame(
        {
            "round_t": grid,
            "mean": np.nanmean(arr, axis=0),
            "std": np.nanstd(arr, axis=0),
            "n": np.sum(np.isfinite(arr), axis=0),
        }
    )


def compute_round_average_profile_real_time(
    annotated_trial_data: pl.DataFrame,
    metric: str,
    bin_s: float = 0.25,
    phase_filter=None,
    round_ids: Optional[List[int]] = None,
) -> pd.DataFrame:
    if metric not in annotated_trial_data.columns:
        return pd.DataFrame()

    df = annotated_trial_data.to_pandas()
    phases = _phase_list(phase_filter)
    if phases is not None and "phase" in df.columns:
        df = df[df["phase"].isin(phases)]
    if round_ids is not None and "round_id" in df.columns:
        df = df[df["round_id"].isin(round_ids)]

    needed = {"timestamp", "round_start", "round_id", metric}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    rs = pd.to_datetime(df["round_start"], utc=True, errors="coerce")
    y = pd.to_numeric(df[metric], errors="coerce")

    work = pd.DataFrame({
        "round_id": df["round_id"],
        "t_since_start_s": (ts - rs).dt.total_seconds(),   # >= 0 during round
        "value": y
    }).dropna()

    if work.empty:
        return pd.DataFrame()

    # Bin by real time from round start
    b = float(bin_s)
    work["t_bin"] = np.round(work["t_since_start_s"] / b) * b

    g = work.groupby("t_bin")["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index().sort_values("t_bin")
    return out.rename(columns={"t_bin": "t_since_start_s"})


def compute_round_curves_for_one_rnti(
    annotated_trial_data: pl.DataFrame,
    metric: str,
    cfg: PlotConfig,
) -> Dict[int, pd.DataFrame]:
    if metric not in annotated_trial_data.columns:
        return {}

    mode = (cfg.round_profile_time_mode or "normalized").lower()
    phases = _phase_list(cfg.round_profile_phase_filter)

    df = annotated_trial_data.to_pandas()
    if phases is not None and "phase" in df.columns:
        df = df[df["phase"].isin(phases)]
    if cfg.round_profile_round_ids is not None and "round_id" in df.columns:
        df = df[df["round_id"].isin(cfg.round_profile_round_ids)]

    if mode == "real_from_round_start":
        needed = {"timestamp", "round_start", "round_id", metric}
        if not needed.issubset(df.columns):
            return {}
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        rs = pd.to_datetime(df["round_start"], utc=True, errors="coerce")
        df["x"] = (ts - rs).dt.total_seconds()
    else:
        # round_t may not exist in compact pipeline; compute if missing.
        if "round_t" not in df.columns:
            needed = {"round_start", "round_end", "timestamp"}
            if not needed.issubset(df.columns):
                return {}
            rs = pd.to_datetime(df["round_start"], utc=True, errors="coerce")
            re = pd.to_datetime(df["round_end"], utc=True, errors="coerce")
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            dur = (re - rs).dt.total_seconds()
            el = (ts - rs).dt.total_seconds()
            df["round_t"] = np.where(dur > 0, el / dur, np.nan)
        df["x"] = pd.to_numeric(df["round_t"], errors="coerce")

    df["value"] = pd.to_numeric(df[metric], errors="coerce")
    work = df.dropna(subset=["round_id", "x", "value"]).copy()
    if work.empty:
        return {}

    curves: Dict[int, pd.DataFrame] = {}
    for rid, grp in work.groupby("round_id"):
        g = grp.sort_values("x").copy()
        x = g["x"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 2:
            continue

        # Keep monotonic x for clean lines.
        keep = np.r_[True, np.diff(x) > 0]
        x, y = x[keep], y[keep]
        if len(x) < 2:
            continue

        curves[int(rid)] = pd.DataFrame({"x": x, "value": y})

    return curves


def _selected_round_ids(available_round_ids: List[int], cfg: PlotConfig) -> List[int]:
    available = set(int(r) for r in available_round_ids)
    if cfg.round_profile_round_ids:
        return [int(r) for r in cfg.round_profile_round_ids if int(r) in available]

    ordered = sorted(available)
    max_curves = int(cfg.round_profile_max_curves)
    if max_curves <= 0:
        return ordered
    return ordered[:max_curves]


def _selected_rntis_for_per_round(ue_dfs: Dict[str, pl.DataFrame], cfg: PlotConfig) -> List[str]:
    all_rntis = sorted(str(k) for k in ue_dfs.keys())
    if not all_rntis:
        return []

    layout = (cfg.round_profile_per_round_rnti_layout or "single").lower()
    if layout == "single":
        target = str(cfg.round_profile_target_rnti) if cfg.round_profile_target_rnti is not None else all_rntis[0]
        return [target] if target in ue_dfs else []

    if cfg.round_profile_target_rntis:
        chosen = [str(r) for r in cfg.round_profile_target_rntis if str(r) in ue_dfs]
        return chosen

    if cfg.round_profile_target_rnti is not None and str(cfg.round_profile_target_rnti) in ue_dfs:
        return [str(cfg.round_profile_target_rnti)]

    return all_rntis


def plot_round_profiles(
    ue_dfs: Dict[str, pl.DataFrame],
    metric: str,
    run_label: str,
    cfg: PlotConfig,
    paired_metric: Optional[str] = None,
):
    """
    Unified round-profile plotter:
      - single metric across devices
      - or UL/DL paired metric across devices
      - optional effective sum line for throughput metrics
    """
    layout = (cfg.round_profile_layout or "same_axes").lower()
    phases = _phase_list(cfg.round_profile_phase_filter)
    phase_suffix = "" if phases is None else f" ({'/'.join(phases)})"

    mode = (cfg.round_profile_time_mode or "normalized").lower()
    curve_mode = (cfg.round_profile_curve_mode or "average").lower()
    xlabel = "Time since round start (s)" if mode == "real_from_round_start" else "Normalized Round Time"

    # -------- per-round raw curves mode (no averaging, no binning) --------
    if curve_mode == "per_round":
        ue_by_rnti = {str(k): v for k, v in ue_dfs.items()}
        if not ue_by_rnti:
            return

        layout_mode = (cfg.round_profile_per_round_rnti_layout or "single").lower()
        selected_rntis = _selected_rntis_for_per_round(ue_by_rnti, cfg)
        if not selected_rntis:
            print(f"[round profile] no matching RNTIs found, available={sorted(ue_by_rnti.keys())}")
            return

        curves_by_rnti: Dict[str, Dict[int, pd.DataFrame]] = {}
        for rnti in selected_rntis:
            curves = compute_round_curves_for_one_rnti(ue_by_rnti[rnti], metric=metric, cfg=cfg)
            if curves:
                curves_by_rnti[rnti] = curves

        if not curves_by_rnti:
            return

        # Keep only requested rounds (if provided) or apply max-curves cap.
        rounds_for_rnti: Dict[str, List[int]] = {}
        for rnti, curves in curves_by_rnti.items():
            chosen = _selected_round_ids(list(curves.keys()), cfg)
            if chosen:
                rounds_for_rnti[rnti] = chosen

        if not rounds_for_rnti:
            print(f"[round profile] requested rounds not found: {cfg.round_profile_round_ids}")
            return

        if paired_metric:
            print("[round profile] per_round mode currently plots only the primary metric")

        if layout_mode == "same_axes" and len(rounds_for_rnti) > 1:
            fig, ax = plt.subplots(figsize=(11, 5))
            rntis = sorted(rounds_for_rnti.keys())
            palette = sns.color_palette("tab10", n_colors=max(1, len(rntis)))
            color_map = {r: palette[i % len(palette)] for i, r in enumerate(rntis)}

            # Legend denotes RNTI color, not round number.
            for rnti in rntis:
                first = True
                for rid in rounds_for_rnti[rnti]:
                    p = curves_by_rnti[rnti][rid]
                    ax.plot(
                        p["x"],
                        p["value"],
                        linewidth=1.4,
                        alpha=0.55,
                        color=color_map[rnti],
                        label=rnti if first else "_nolegend_",
                    )
                    first = False

            rounds_txt = "all" if cfg.round_profile_round_ids is None else ",".join(str(r) for r in cfg.round_profile_round_ids)
            title = f"{metric} round curves [{run_label}] [rounds={rounds_txt}]{phase_suffix}"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(True)
            ax.legend(title="RNTI", ncol=2, fontsize=8)

            fig.tight_layout()
            if cfg.output_dir is not None:
                plt.savefig(cfg.output_dir / f"{_safe_filename(title)}.pdf")
            if cfg.show_plots:
                plt.show()
            else:
                plt.close(fig)
            return

        # single or separate_figures mode
        for rnti, chosen_rounds in rounds_for_rnti.items():
            fig, ax = plt.subplots(figsize=(11, 5))
            colors = sns.color_palette("tab20", n_colors=max(1, len(chosen_rounds)))
            for i, rid in enumerate(chosen_rounds):
                p = curves_by_rnti[rnti][rid]
                ax.plot(p["x"], p["value"], linewidth=1.6, color=colors[i % len(colors)], label=f"round {rid}")

            rounds_txt = "all" if cfg.round_profile_round_ids is None else ",".join(str(r) for r in cfg.round_profile_round_ids)
            title = f"{metric} round curves [{run_label}] [RNTI {rnti}] [rounds={rounds_txt}]{phase_suffix}"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.grid(True)
            ax.legend(ncol=2, fontsize=8)

            fig.tight_layout()
            if cfg.output_dir is not None:
                plt.savefig(cfg.output_dir / f"{_safe_filename(title)}.pdf")
            if cfg.show_plots:
                plt.show()
            else:
                plt.close(fig)
        return

    # -------- averaged profile mode --------
    prof_a, prof_b = {}, {}
    for dev, df in ue_dfs.items():
        if "round_id" not in df.columns:
            continue
        if cfg.round_profile_time_mode == "real_from_round_start":
            p1 = compute_round_average_profile_real_time(df, metric, bin_s=cfg.round_profile_bin_s, phase_filter=cfg.round_profile_phase_filter, round_ids=cfg.round_profile_round_ids)
        else:
            p1 = compute_round_average_profile(df, metric, n_points=cfg.round_profile_points, phase_filter=cfg.round_profile_phase_filter, round_ids=cfg.round_profile_round_ids)

        if not p1.empty:
            prof_a[str(dev)] = p1
        if paired_metric:
            if cfg.round_profile_time_mode == "real_from_round_start":
                p2 = compute_round_average_profile_real_time(df, paired_metric, bin_s=cfg.round_profile_bin_s, phase_filter=cfg.round_profile_phase_filter, round_ids=cfg.round_profile_round_ids)
            else:
                p2 = compute_round_average_profile(df, paired_metric, n_points=cfg.round_profile_points, phase_filter=cfg.round_profile_phase_filter, round_ids=cfg.round_profile_round_ids)
            if not p2.empty:
                prof_b[str(dev)] = p2

    if not prof_a and not prof_b:
        return
    
    xcol = "t_since_start_s" if mode == "real_from_round_start" else "round_t"

    # -------- paired UL/DL mode --------
    if paired_metric:
        if layout == "subplots":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            for dev, p in prof_a.items():
                ax1.plot(p[xcol], p["mean"], linewidth=2, label=dev)
                if cfg.round_profile_ci_bands and {"mean", "std", "n"}.issubset(p.columns):
                    hw = _ci_halfwidth(p["std"].to_numpy(float), p["n"].to_numpy(float), cfg.round_profile_ci_level)
                    m = p["mean"].to_numpy(float)
                    x = p[xcol].to_numpy(float)
                    ax.fill_between(x, m - hw, m + hw, alpha=0.18)
            for dev, p in prof_b.items():
                ax2.plot(p[xcol], p["mean"], linewidth=2, label=dev)
                if cfg.round_profile_ci_bands and {"mean", "std", "n"}.issubset(p.columns):
                    hw = _ci_halfwidth(p["std"].to_numpy(float), p["n"].to_numpy(float), cfg.round_profile_ci_level)
                    m = p["mean"].to_numpy(float)
                    x = p[xcol].to_numpy(float)
                    ax.fill_between(x, m - hw, m + hw, alpha=0.18)
            
            

            ax1.set_ylabel(metric)
            ax2.set_ylabel(paired_metric)
            ax2.set_xlabel(xlabel)
            ax1.set_title(f"{metric} round profile [{xlabel}]{phase_suffix}")
            title = f"{paired_metric} round profile [{xlabel}]{phase_suffix}"
            ax2.set_title(title)
            ax1.grid(True); ax2.grid(True)
            ax1.legend(ncol=2, fontsize=8); ax2.legend(ncol=2, fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
            devices = sorted(set(list(prof_a.keys()) + list(prof_b.keys())))
            colors = sns.color_palette("tab10", n_colors=max(1, len(devices)))
            c = {d: colors[i % len(colors)] for i, d in enumerate(devices)}

            for dev, p in prof_a.items():
                ax.plot(p[xcol], p["mean"], "-", color=c[dev], linewidth=2, label=f"{dev} UL")
                if cfg.round_profile_ci_bands and {"mean", "std", "n"}.issubset(p.columns):
                    hw = _ci_halfwidth(p["std"].to_numpy(float), p["n"].to_numpy(float), cfg.round_profile_ci_level)
                    m = p["mean"].to_numpy(float)
                    x = p[xcol].to_numpy(float)
                    ax.fill_between(x, m - hw, m + hw, alpha=0.9,color=c[dev])
            for dev, p in prof_b.items():
                ax.plot(p[xcol], p["mean"], "--", color=c[dev], linewidth=2, label=f"{dev} DL")
            
                if cfg.round_profile_ci_bands and {"mean", "std", "n"}.issubset(p.columns):
                    hw = _ci_halfwidth(p["std"].to_numpy(float), p["n"].to_numpy(float), cfg.round_profile_ci_level)
                    m = p["mean"].to_numpy(float)
                    x = p[xcol].to_numpy(float)
                    ax.fill_between(x, m - hw, m + hw, alpha=0.9,color=c[dev])

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"{metric} / {paired_metric}")
            title = f"UL-DL round profile [{xlabel}]{phase_suffix}"
            ax.set_title(title)
            ax.grid(True)
            ax.legend(ncol=2, fontsize=8)

        fig.tight_layout()
        if cfg.output_dir is not None:
            plt.savefig(cfg.output_dir / f"{_safe_filename(title)}.pdf")
        if cfg.show_plots:
            plt.show()
        else:
            plt.close(fig)
        return

    # -------- single-metric mode --------
    fig, ax = plt.subplots(figsize=(11, 5))
    effective = None
    grid = None

    for dev, p in prof_a.items():
        ax.plot(p[xcol], p["mean"], linewidth=2, label=dev)

        if cfg.round_profile_ci_bands and {"mean", "std", "n"}.issubset(p.columns):
            hw = _ci_halfwidth(p["std"].to_numpy(float), p["n"].to_numpy(float), cfg.round_profile_ci_level)
            m = p["mean"].to_numpy(float)
            x = p[xcol].to_numpy(float)
            ax.fill_between(x, m - hw, m + hw, alpha=0.18)

        if cfg.round_profile_error_bars:
            step = max(1, int(cfg.round_profile_errorbar_step))
            idx = np.arange(0, len(p), step)
            ax.errorbar(
                p[xcol].to_numpy()[idx],
                p["mean"].to_numpy()[idx],
                yerr=p["std"].to_numpy()[idx],
                fmt="none",
                alpha=0.35,
                capsize=2,
            )

        if cfg.round_profile_include_effective_sum and metric in THROUGHPUT_METRICS:
            if effective is None:
                effective = np.zeros(len(p), dtype=float)
                grid = p[xcol].to_numpy(dtype=float)
            effective += p["mean"].to_numpy(dtype=float)

    if effective is not None:
        if cfg.round_profile_effective_secondary_axis:
            ax2 = ax.twinx()
            ax2.plot(grid, effective, "k--", linewidth=2, label="effective total")
            ax2.set_ylabel(f"effective {metric}")
        else:
            ax.plot(grid, effective, "k--", linewidth=2, label="effective total")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    title = f"{metric} round profile [{run_label}]{phase_suffix}"
    ax.set_title(title)
    ax.grid(True)
    ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    if cfg.output_dir is not None:
        plt.savefig(cfg.output_dir / f"{_safe_filename(title)}.pdf")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)

# =========================
# Plotting
# =========================

def _distribution_df(ue_dfs: Dict[str, pl.DataFrame], metric: str, thresholds: Dict[str, float], paired_metric: Optional[str]):
    rows = []
    for rnti, df in ue_dfs.items():
        for v in _metric_values(df, metric, thresholds):
            row = {"device": str(rnti), "value": v, "series": metric}
            if paired_metric:
                row["direction"] = "UL"
            rows.append(row)

        if paired_metric and paired_metric in df.columns:
            for v in _metric_values(df, paired_metric, thresholds):
                rows.append({"device": str(rnti), "value": v, "series": paired_metric, "direction": "DL"})

    return pd.DataFrame(rows)


def plot_metric(ue_dfs: Dict[str, pl.DataFrame], metric: str, run_label: str, cfg: PlotConfig, paired_metric: Optional[str] = None):
    thresholds = _norm_thresholds(cfg.min_thresholds)

    if "time" in cfg.plot_mode:
        fig, ax = plt.subplots(figsize=(10, 6))
        rows = []
        for rnti, df in ue_dfs.items():
            if metric not in df.columns:
                continue
            pdf = df.select(["timestamp", metric]).to_pandas().dropna(subset=[metric])
            t = thresholds.get(metric)
            if t is not None:
                pdf[metric] = pd.to_numeric(pdf[metric], errors="coerce")
                pdf = pdf[pdf[metric] >= t]
            
            if cfg.use_relative_time and not pdf.empty:
                ts = pd.to_datetime(pdf["timestamp"], utc=True, errors="coerce")
                pdf["relative_time_s"] = (ts - ts.min()).dt.total_seconds()
            
            pdf = pdf.iloc[cfg.pts_offset: cfg.pts_offset + cfg.pts_to_plot]
            for _, r in pdf.iterrows():
                rows.append({
                    "timestamp": r["timestamp"],
                    "relative_time_s": r.get("relative_time_s", np.nan),
                    "value": r[metric],
                    "device": str(rnti),
                })

        if not rows:
            plt.close(fig)
            return

        plot_df = pd.DataFrame(rows)
        # if cfg.smoothing:
        #     plot_df['value'] = plot_df.transform(lambda s: s.rolling(10, min_periods=1).mean())
        x_col = "relative_time_s" if cfg.use_relative_time else "timestamp"
        try: 
            sns.scatterplot(data=plot_df, x=x_col, y="value", hue="device", alpha=0.8, ax=ax)
        except ValueError:
            print(f'x_col: {x_col}, plot_df: {plot_df}')
            sns.scatterplot(data=plot_df, x='timestamp', y="value", hue="device", alpha=0.8, ax=ax)
        ax.set_xlabel("Relative Time (s)" if cfg.use_relative_time else "Timestamp")
        # sns.scatterplot(data=plot_df, x="timestamp", y="value", hue="device", alpha=0.8, ax=ax)
        # ax.plot(plot_df['timestamp'], plot_df['value'])
        title = f"{metric} over time [{run_label}]"
        ax.set_title(title)
        ax.grid(True)
        if cfg.save_plots:
            plt.savefig(cfg.output_dir / f"{_safe_filename(title)}.pdf")
        if cfg.show_plots:
            plt.show()
        else:
            plt.close(fig)

    # distribution mode
    if "dist" in cfg.plot_mode:
        df = _distribution_df(ue_dfs, metric, thresholds, paired_metric)
        if df.empty:
            return

        kind_map = {"violin": "violin", "box": "box", "count": "count", "bar": "bar", "kde": "bar"}
        kind = kind_map.get(cfg.distribution_plot_type, "violin")

        kwargs = {"data": df, "x": "device", "kind": kind, "height": 6, "aspect": 1.8}
        if paired_metric:
            kwargs["hue"] = "direction"
        if kind in {"violin", "box", "bar"}:
            kwargs["y"] = "value"
        if kind == "violin":
            kwargs.update({"inner": "quart", "cut": 0})
            if paired_metric:
                kwargs.update({"split": True, "gap": 0.1})

        g = sns.catplot(**kwargs)
        ax = g.ax
        title = f"{metric} distribution [{run_label}]" if not paired_metric else f"{metric} vs {paired_metric} [{run_label}]"
        ax.set_title(title)
        ax.grid(True)

        if cfg.save_plots:
            plt.savefig(cfg.output_dir / f"{_safe_filename(title)}.pdf")
        if cfg.show_plots:
            plt.show()
        else:
            plt.close(g.figure)


# =========================
# Experiment selection + run labels
# =========================

def build_experiment_index(root_dir: Path) -> List[dict]:
    exps = []
    for exp in root_dir.iterdir():
        if exp.is_dir():
            exps.append({"path": exp, **parse_experiment_name(exp.name)})
    return exps


def filter_experiments(experiments: List[dict], filters: Dict[str, Any]) -> List[dict]:
    out = []
    for exp in experiments:
        keep = True
        for k, v in filters.items():
            ev = exp.get(k)

            # Special-case bandwidth so "20" matches "20 MHz"
            if k == "bandwidth":
                candidates = {str(ev)} if ev is not None else set()
                bw_raw = exp.get("bandwidth_raw")
                if bw_raw is not None:
                    candidates.add(str(bw_raw))
                    candidates.add(f"{bw_raw} MHz")

                if isinstance(v, list):
                    if not any(str(x) in candidates for x in v):
                        keep = False
                        break
                else:
                    if str(v) not in candidates:
                        keep = False
                        break
                continue

            if ev is None:
                continue
            if isinstance(v, list) and ev not in v:
                keep = False
                break
            if not isinstance(v, list) and ev != v:
                keep = False
                break
        if keep:
            out.append(exp)
    return out


def run_label_for(exp: dict, sweep: Optional[str]) -> str:
    # Iperf experiments: readable composite label
    if "location" in exp and "tdd" in exp and ("bandwidth" in exp or "bandwidth_raw" in exp):
        bw = exp.get("bandwidth", f"{exp.get('bandwidth_raw', 'NA')} MHz")
        loc = exp.get("location", "unknown")
        tdd = exp.get("tdd", "NA")
        return f"{loc} | {bw} | {tdd}"

    # Original behavior for FedAvg experiments
    if sweep:
        return str(format_sweep_label(sweep, exp[sweep], exp))
    return exp["path"].name


# =========================
# Pipeline
# =========================

def summarize_throughput(ue_dfs: Dict[str, pl.DataFrame], run_label: str, thresholds: Dict[str, float]):
    rows = []
    for rnti, df in ue_dfs.items():
        if not {"ul_throughput_mbps", "dl_throughput_mbps"}.issubset(df.columns):
            continue
        ul = _metric_values(df, "ul_throughput_mbps", thresholds)
        dl = _metric_values(df, "dl_throughput_mbps", thresholds)
        rows.append({
            "run_id": run_label,
            "device": rnti,
            "avg_ul_throughput_mbps": pd.Series(ul).mean() if ul else None,
            "std_ul_throughput_mbps": pd.Series(ul).std() if len(ul) > 1 else None,
            "avg_dl_throughput_mbps": pd.Series(dl).mean() if dl else None,
            "std_dl_throughput_mbps": pd.Series(dl).std() if len(dl) > 1 else None,
        })
    return rows


def run_analysis(exp: dict, cfg: PlotConfig):
    label = run_label_for(exp, cfg.sweep)
    thresholds = _norm_thresholds(cfg.min_thresholds)

    if cfg.dataset_type.lower() == "iperf":
        ue_dfs = load_iperf_experiment_data(
            exp["path"],
            cfg.metrics,
            cid_filter=cfg.cid_filter,
            direction_filter=cfg.direction_filter,
        )
    else:
        ue_dfs = load_experiment_data(exp, cfg.metrics)
    
    if not ue_dfs:
        print(f"[{label}] no UE data")
        return []
    
    if cfg.dataset_type.lower() != "iperf":
        ue_dfs, _rounds = apply_round_processing(exp["path"], ue_dfs, cfg)

    processed = set()
    for metric in cfg.metrics:
        if metric in processed:
            continue

        paired = None
        if cfg.pair_ul_dl and metric.startswith("ul_"):
            candidate = "dl_" + metric[3:]
            if candidate in cfg.metrics:
                paired = candidate
                processed.add(candidate)

        m, _s = _compute_mean_std(ue_dfs, metric, thresholds)
        if paired:
            pm, _ps = _compute_mean_std(ue_dfs, paired, thresholds)
            if m is None and pm is None:
                continue
        else:
            if m is None:
                continue

        plot_metric(ue_dfs, metric, label, cfg, paired_metric=paired)
        if cfg.round_profiles_enabled:
            plot_round_profiles(
                ue_dfs,
                metric=metric,
                paired_metric=paired,
                run_label=label,
                cfg=cfg,
            )
        processed.add(metric)

    print(f"[{label}] complete")
    return summarize_throughput(ue_dfs, label, thresholds)


def run_pipeline(cfg: PlotConfig):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.dataset_type.lower() == "iperf":
        if cfg.iperf_root_dir is None:
            raise ValueError("cfg.iperf_root_dir must be set when dataset_type='iperf'")
        exps = build_iperf_experiment_index(cfg.iperf_root_dir)
    else:
        exps = build_experiment_index(cfg.data_dir)

    exps = filter_experiments(exps, cfg.filters)
    if cfg.sweep and cfg.dataset_type.lower() != "iperf":
        exps = sort_experiments_by_sweep(exps, cfg.sweep)

    if not exps:
        print("No experiments matched filters")
        return

    all_rows = []
    for exp in exps:
        all_rows.extend(run_analysis(exp, cfg))

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_fp = cfg.output_dir / "throughput_summary_by_device.csv"
        df.to_csv(out_fp, index=False)
        print(f"Saved throughput summary: {out_fp}")


def main():
    cfg = PlotConfig(
        data_dir=Path("/Users/kmcomer/Documents/5G Experiment Data/FedAvg/"),
        output_dir=Path.cwd() / "phys_layer_plots",
        filters={
            # "bandwidth": "20 MHz",
            # "distribution": "dirichlet",
            # "tdd": "2-7",
            # "nodes": "9N",
            "rank": "2x2",
            # "congestion": False,
        },
        sweep="network",
        # metrics=["ulMcs", "dlMcs", "puschSnr", "rssi", "ul_throughput_mbps", "dl_throughput_mbps", "phr", "ulBler", "dlBler", "ul_shannon", "dl_shannon", "ul_3gpp", "dl_3gpp"],
        metrics=["ul_throughput_mbps", "dl_throughput_mbps"],#["ul_shannon", "dl_shannon", "ul_3gpp", "dl_3gpp", "ul_shannon_sinr", "dl_shannon_sinr"],
        # min_thresholds={"ul_throughput_mbps": 0.01, "dl_throughput_mbps": 0.01},
        filter_rounds=True,
        annotate_phases=True,
        pair_ul_dl=False,
        plot_mode=["time"],
        smoothing=False,
        pts_to_plot = 1000,
        pts_offset = 0,
        distribution_plot_type="box",
        show_plots=True,
        round_profiles_enabled=True,
        # round_profile_round_ids=[2,3,4,5,6],
        round_profile_include_effective_sum = False,
        round_profile_points = 10000,
        use_relative_time=False,
        # round_profile_round_ids=[3,77,180],  # choose exact rounds in per_round mode
        round_profile_curve_mode = "average",   # "average" | "per_round"
        round_profile_target_rnti = None,        # e.g. "65" for per_round mode
        round_profile_target_rntis = None,       # e.g. ["65", "66"]; None means all in multi modes
        round_profile_per_round_rnti_layout = "single",  # "single" | "same_axes" | "separate_figures"
        round_profile_max_curves = 12,
        round_profile_time_mode = "real_from_round_start",   # "normalized" | "real_from_round_start"
        round_profile_ci_bands = True,
        round_profile_ci_level = 0.95,
        round_profile_bin_s = 0.1,   # for real-time mode binning
    )

    # ["segment", "pucchSnr", "ranUeId", "dlBytes", "dlMcs", "ulQm", "rsrp", "ueId", "amfUeId", "dlQm",
    #  "ulMcs", "ulBler", "puschSnr", "dlBler", "ulBytes", "pmi", "rssi", "cqi", "inSync", "ri", "phr",
    #  "pcmax", "sinr", "rsrq"]

    # cfg = PlotConfig(
    #     data_dir=Path("/unused/for/iperf"),
    #     iperf_root_dir=Path("/Users/kmcomer/Documents/5G Experiment Data"),
    #     output_dir=Path.cwd() / "iperf_plots",
    #     dataset_type="iperf",
    #     filters={
    #         "bandwidth": ["100",],   # also accepts "20 MHz"
    #         "tdd": ["2-7"],
    #         "location": ["all_connected"],
    #     },
    #     cid_filter=["1", "2", "3", "4", "5", "6"],
    #     direction_filter=["UL", "DL"], # ["UL", "DL"]
    #     metrics=["rssi", "ul_throughput_mbps", "dl_throughput_mbps", "rsrp"],
    #     plot_mode="time",
    #     use_relative_time=False,
    #     pair_ul_dl=False,
    #     show_plots=True,
    #     save_plots=True,
    # )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()


# round_profiles_enabled: bool = False
#     round_profile_points: int = 100000
#     round_profile_phase_filter: Optional[List[str]] = None
#     round_profile_round_ids: Optional[List[int]] = None
#     round_profile_layout: str = "same_axes" # use 'subplots' for stacked UL/DL panels
#     round_profile_error_bars: bool = False
#     round_profile_errorbar_step: int = 10
#     round_profile_include_effective_sum: bool = True
#     round_profile_effective_secondary_axis: bool = True