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
}


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
    round_profile_points: int = 1000
    round_profile_phase_filter: Optional[List[str]] = None
    round_profile_round_ids: Optional[List[int]] = None
    round_profile_layout: str = "same_axes" # use 'subplots' for stacked UL/DL panels
    round_profile_error_bars: bool = False
    round_profile_errorbar_step: int = 10
    round_profile_include_effective_sum: bool = True
    round_profile_effective_secondary_axis: bool = True

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


# =========================
# Data loading
# =========================

def _metrics_to_load(metrics: List[str]) -> List[str]:
    base = [m for m in metrics if m not in THROUGHPUT_METRICS]
    if any(m in THROUGHPUT_METRICS for m in metrics):
        base += ["ulBytes", "dlBytes"]
    return list(dict.fromkeys(base))


def _read_joined_csv(main_fp: str, secondary_fp: Optional[str], columns: Optional[List[str]] = None) -> pl.DataFrame:
    if secondary_fp:
        left = pl.read_csv(main_fp, columns=["timestamp"], try_parse_dates=True).with_row_index("segment")
        right = pl.read_csv(secondary_fp, columns=(["segment"] + columns) if columns else None)
        return left.join(right, on="segment", how="right")
    return pl.read_csv(main_fp, columns=(["timestamp"] + columns) if columns else None, try_parse_dates=True)


def _add_throughput(df: pl.DataFrame) -> pl.DataFrame:
    needed = {"timestamp", "ulBytes", "dlBytes"}
    if not needed.issubset(df.columns):
        return df

    return (
        df.with_columns(
            ul_num=pl.col("ulBytes").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
            dl_num=pl.col("dlBytes").cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False),
        )
        .sort("timestamp")
        .with_columns(
            dt=pl.col("timestamp").diff().dt.total_seconds(),
            dul=pl.col("ul_num").diff(),
            ddl=pl.col("dl_num").diff(),
        )
        .with_columns(
            ul_throughput_mbps=pl.when((pl.col("dt") > 0) & (pl.col("dul") >= 0)).then((pl.col("dul") * 8 / 1_000_000) / pl.col("dt")).otherwise(None),
            dl_throughput_mbps=pl.when((pl.col("dt") > 0) & (pl.col("ddl") >= 0)).then((pl.col("ddl") * 8 / 1_000_000) / pl.col("dt")).otherwise(None),
        )
        .drop("ul_num", "dl_num")
    )


def load_experiment_data(exp_path: Path, metrics: List[str]) -> Dict[str, pl.DataFrame]:
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
        out[rnti] = _add_throughput(df)

    return out

# iperf data loading
def build_iperf_experiment_index(root_dir: Path) -> List[dict]:
    exps = []
    location_dirs = [
        ("normal", root_dir / "iperf_normal_locations"),
        ("fair", root_dir / "iperf_fair"),
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

    arr = np.vstack(aligned)
    return pd.DataFrame(
        {"round_t": grid, "mean": np.nanmean(arr, axis=0), "std": np.nanstd(arr, axis=0)}
    )


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

    # build profiles once
    prof_a, prof_b = {}, {}
    for dev, df in ue_dfs.items():
        if "round_id" not in df.columns:
            continue
        p1 = compute_round_average_profile(df, metric, n_points=cfg.round_profile_points, phase_filter=cfg.round_profile_phase_filter, round_ids=cfg.round_profile_round_ids)
        if not p1.empty:
            prof_a[str(dev)] = p1
        if paired_metric:
            p2 = compute_round_average_profile(df, paired_metric, n_points=cfg.round_profile_points, phase_filter=cfg.round_profile_phase_filter, round_ids=cfg.round_profile_round_ids)
            if not p2.empty:
                prof_b[str(dev)] = p2

    if not prof_a and not prof_b:
        return

    # -------- paired UL/DL mode --------
    if paired_metric:
        if layout == "subplots":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
            for dev, p in prof_a.items():
                ax1.plot(p["round_t"], p["mean"], linewidth=2, label=dev)
            for dev, p in prof_b.items():
                ax2.plot(p["round_t"], p["mean"], linewidth=2, label=dev)

            ax1.set_ylabel(metric)
            ax2.set_ylabel(paired_metric)
            ax2.set_xlabel("Normalized Round Time")
            ax1.set_title(f"{metric} round profile [{run_label}]{phase_suffix}")
            title = f"{paired_metric} round profile [{run_label}]{phase_suffix}"
            ax2.set_title(title)
            ax1.grid(True); ax2.grid(True)
            ax1.legend(ncol=2, fontsize=8); ax2.legend(ncol=2, fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(11, 5))
            devices = sorted(set(list(prof_a.keys()) + list(prof_b.keys())))
            colors = sns.color_palette("tab10", n_colors=max(1, len(devices)))
            c = {d: colors[i % len(colors)] for i, d in enumerate(devices)}

            for dev, p in prof_a.items():
                ax.plot(p["round_t"], p["mean"], "-", color=c[dev], linewidth=2, label=f"{dev} UL")
            for dev, p in prof_b.items():
                ax.plot(p["round_t"], p["mean"], "--", color=c[dev], linewidth=2, label=f"{dev} DL")

            ax.set_xlabel("Normalized Round Time")
            ax.set_ylabel(f"{metric} / {paired_metric}")
            title = f"UL-DL round profile [{run_label}]{phase_suffix}"
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
        ax.plot(p["round_t"], p["mean"], linewidth=2, label=dev)

        if cfg.round_profile_error_bars:
            step = max(1, int(cfg.round_profile_errorbar_step))
            idx = np.arange(0, len(p), step)
            ax.errorbar(
                p["round_t"].to_numpy()[idx],
                p["mean"].to_numpy()[idx],
                yerr=p["std"].to_numpy()[idx],
                fmt="none",
                alpha=0.35,
                capsize=2,
            )

        if cfg.round_profile_include_effective_sum and metric in THROUGHPUT_METRICS:
            if effective is None:
                effective = np.zeros(len(p), dtype=float)
                grid = p["round_t"].to_numpy(dtype=float)
            effective += p["mean"].to_numpy(dtype=float)

    if effective is not None:
        if cfg.round_profile_effective_secondary_axis:
            ax2 = ax.twinx()
            ax2.plot(grid, effective, "k--", linewidth=2, label="effective total")
            ax2.set_ylabel(f"effective {metric}")
        else:
            ax.plot(grid, effective, "k--", linewidth=2, label="effective total")

    ax.set_xlabel("Normalized Round Time")
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

    if cfg.plot_mode == "time":
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
    elif cfg.plot_mode == "dist":
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
        ue_dfs = load_experiment_data(exp["path"], cfg.metrics)
    
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
            "bandwidth": "100 MHz",
            "rank": "2x2",
            "distribution": "dirichlet",
            "congestion": False,
            "tdd": "7-2",
            "nodes": "6N",
        },
        sweep="network",
        metrics=["rssi", "ul_throughput_mbps", "dl_throughput_mbps"],
        min_thresholds={"ul_throughput_mbps": 0.01, "dl_throughput_mbps": 0.01},
        filter_rounds=True,
        annotate_phases=True,
        pair_ul_dl=True,
        plot_mode=None,
        smoothing=True,
        pts_to_plot = 100,
        pts_offset = 1000,
        distribution_plot_type="box",
        show_plots=True,
        round_profiles_enabled=True,
    )
    # cfg = PlotConfig(
    #     data_dir=Path("/unused/for/iperf"),
    #     iperf_root_dir=Path("/Users/kmcomer/Documents/5G Experiment Data"),
    #     output_dir=Path.cwd() / "iperf_plots",
    #     dataset_type="iperf",
    #     filters={
    #         "bandwidth": ["20", "40", "80"],   # also accepts "20 MHz"
    #         "tdd": ["7-2", "5-4"],
    #         "location": ["normal", "fair"],
    #     },
    #     cid_filter=["1", "2", "3", "4", "5", "6"],
    #     direction_filter=["UL"], # ["UL", "DL"]
    #     metrics=["rssi", "ul_throughput_mbps", "dl_throughput_mbps"],
    #     plot_mode="dist",
    #     use_relative_time=False,
    #     pair_ul_dl=False,
    #     show_plots=True,
    #     save_plots=True,
    # )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()