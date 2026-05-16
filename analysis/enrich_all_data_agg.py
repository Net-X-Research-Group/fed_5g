from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
from tqdm import tqdm

from telemetry_parsing import gather_metrics_by_rnti
from telemetry_plotting import _add_throughput, PlotConfig, apply_round_processing


# Metrics produced by telemetry_plotting._add_throughput
UL_METRICS = ["ul_throughput_mbps", "ul_3gpp", "ul_shannon", "ul_shannon_sinr"]
DL_METRICS = ["dl_throughput_mbps", "dl_3gpp", "dl_shannon", "dl_shannon_sinr"]
ALL_METRICS = UL_METRICS + DL_METRICS

# PHY CSV columns can contain mixed integer-looking and decimal-looking values
# (e.g., 5399 and 5399.0), so force Float64 to avoid parser type conflicts.
PHY_SCHEMA_OVERRIDES = {
    "run_id": pl.Utf8,
    "rnti": pl.Utf8,
    "ulBytes": pl.Float64,
    "dlBytes": pl.Float64,
    "ulQm": pl.Float64,
    "dlQm": pl.Float64,
    "ulMcs": pl.Float64,
    "dlMcs": pl.Float64,
    "puschSnr": pl.Float64,
    "sinr": pl.Float64,
}


def _safe_numeric_expr(col_name: str) -> pl.Expr:
    return pl.col(col_name).cast(pl.Utf8).str.replace_all(",", "").cast(pl.Float64, strict=False)


def _norm_run_id_value(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return ""

    # Preserve exact integer-like IDs without float conversion.
    if re.fullmatch(r"\d+", s):
        return s.lstrip("0") or "0"
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0].lstrip("0") or "0"

    # Handle scientific notation text safely.
    try:
        d = Decimal(s)
        if d == d.to_integral_value():
            return format(d.to_integral_value(), "f")
    except (InvalidOperation, ValueError):
        pass

    return s


def _canonical_run_id_col(df: pl.DataFrame, col_name: str = "run_id") -> pl.DataFrame:
    return df.with_columns(
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .map_elements(_norm_run_id_value, return_dtype=pl.Utf8)
        .alias(col_name)
    )


def _pick_col(df: pl.DataFrame, options: List[str], what: str) -> str:
    cols = set(df.columns)
    for c in options:
        if c in cols:
            return c
    raise ValueError(f"Could not find {what}. Tried: {options}")


def _ensure_timestamp_datetime(df: pl.DataFrame, col_name: str = "timestamp") -> pl.DataFrame:
    dt_type = df.schema.get(col_name)
    if dt_type in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")):
        return df.with_columns(pl.col(col_name).dt.replace_time_zone("UTC"))

    if dt_type in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Float64, pl.Float32):
        med = (
            df.select(pl.col(col_name).cast(pl.Float64, strict=False).median().alias("m"))
            .to_series()
            .item()
        )
        if med is not None and float(med) > 1e12:
            return df.with_columns(pl.from_epoch(pl.col(col_name).cast(pl.Int64, strict=False), time_unit="ms").dt.replace_time_zone("UTC").alias(col_name))
        return df.with_columns(pl.from_epoch(pl.col(col_name).cast(pl.Int64, strict=False), time_unit="s").dt.replace_time_zone("UTC").alias(col_name))

    # parse text timestamps
    return df.with_columns(
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias(col_name)
    )


def add_throughput_columns_phy(df: pl.DataFrame, ref_csv: Path, phy_device_col: str) -> pl.DataFrame:
    req = [
        "timestamp",
        "ulBytes",
        "dlBytes",
        "ulQm",
        "dlQm",
        "ulMcs",
        "dlMcs",
        "puschSnr",
        "sinr",
        "bandwidth",
        "tdd",
        "run_id",
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"PHY input missing required columns: {miss}")

    df = _canonical_run_id_col(df, "run_id")
    expected_cols = list(df.columns) + [m for m in ALL_METRICS if m not in df.columns]

    out_parts: List[pl.DataFrame] = []
    for g in df.partition_by(["run_id", phy_device_col], as_dict=False, maintain_order=True):
        g = _ensure_timestamp_datetime(g, "timestamp")
        g = g.drop_nulls("timestamp").sort("timestamp")
        if g.height == 0:
            continue

        bw = g.select(pl.col("bandwidth").drop_nulls().first()).to_series().item()
        tdd = g.select(pl.col("tdd").drop_nulls().first()).to_series().item()
        bw = str(bw) if bw is not None else "20 MHz"
        tdd = str(tdd) if tdd is not None else "2-2"
        exp = {"bandwidth": bw, "tdd": tdd, "_num_users": 1}

        gpl_out = _add_throughput(g, exp)

        ul_nonnull = (
            gpl_out.select(pl.col("ul_throughput_mbps").is_not_null().sum().alias("n")).to_series().item()
            if "ul_throughput_mbps" in gpl_out.columns
            else 0
        )
        dl_nonnull = (
            gpl_out.select(pl.col("dl_throughput_mbps").is_not_null().sum().alias("n")).to_series().item()
            if "dl_throughput_mbps" in gpl_out.columns
            else 0
        )

        # Keep _add_throughput's theory/Shannon outputs; recompute actual throughput if null.
        if int(ul_nonnull or 0) == 0 or int(dl_nonnull or 0) == 0:
            gpl_out = gpl_out.with_columns(
                dt_s=pl.col("timestamp").diff().dt.total_milliseconds() / 1000.0,
                ul_num=_safe_numeric_expr("ulBytes"),
                dl_num=_safe_numeric_expr("dlBytes"),
            ).with_columns(
                dul=pl.col("ul_num").diff(),
                ddl=pl.col("dl_num").diff(),
            ).with_columns(
                pl.when((pl.col("dt_s") > 0) & (pl.col("dul") >= 0))
                .then((pl.col("dul") * 8.0 / 1_000_000.0) / pl.col("dt_s"))
                .otherwise(None)
                .alias("ul_throughput_mbps"),
                pl.when((pl.col("dt_s") > 0) & (pl.col("ddl") >= 0))
                .then((pl.col("ddl") * 8.0 / 1_000_000.0) / pl.col("dt_s"))
                .otherwise(None)
                .alias("dl_throughput_mbps"),
            ).drop(["dt_s", "ul_num", "dl_num", "dul", "ddl"])

        missing_out = [c for c in expected_cols if c not in gpl_out.columns]
        if missing_out:
            gpl_out = gpl_out.with_columns([pl.lit(None).alias(c) for c in missing_out])
        gpl_out = gpl_out.select(expected_cols)

        out_parts.append(gpl_out)

    if not out_parts:
        raise ValueError("No PHY rows remained after timestamp parsing.")

    return pl.concat(out_parts, how="vertical_relaxed")


def build_phy_input_from_trials(fl_df: pl.DataFrame, fedavg_root: Path) -> pl.DataFrame:
    required = {"run_id", "bandwidth", "tdd"}
    missing = required - set(fl_df.columns)
    if missing:
        raise ValueError(f"FL input missing columns needed to build PHY input: {sorted(missing)}")

    meta = (
        _canonical_run_id_col(fl_df.select(["run_id", "bandwidth", "tdd"]), "run_id")
        .group_by("run_id")
        .agg(
            pl.col("bandwidth").drop_nulls().first().alias("bandwidth"),
            pl.col("tdd").drop_nulls().first().alias("tdd"),
        )
    )

    cfg = PlotConfig(
        data_dir=Path("/tmp"),
        output_dir=Path("/tmp"),
        filters={},
        metrics=["ul_throughput_mbps"],
        filter_rounds=True,
        annotate_phases=True,
        show_plots=False,
        save_plots=False,
    )

    parts: List[pl.DataFrame] = []
    for run_id, bw, tdd in meta.iter_rows():
        if not run_id:
            continue
        matches = sorted(fedavg_root.glob(f"{run_id}_*"))
        if not matches:
            continue
        run_dir = matches[0]
        phys_dir = run_dir / "phys_layer"
        if not phys_dir.exists() or not (phys_dir / "common.csv").exists():
            continue

        try:
            ue_dfs = gather_metrics_by_rnti(
                phys_dir,
                metrics=["ulBytes", "dlBytes", "ulQm", "dlQm", "ulMcs", "dlMcs", "puschSnr", "sinr"],
            )
        except (FileNotFoundError, OSError):
            continue

        if not ue_dfs:
            continue

        try:
            ue_proc, _ = apply_round_processing(run_dir, ue_dfs, cfg)
        except Exception:
            ue_proc = ue_dfs

        for rnti, df in ue_proc.items():
            cols = [
                c
                for c in ["timestamp", "round_id", "ulBytes", "dlBytes", "ulQm", "dlQm", "ulMcs", "dlMcs", "puschSnr", "sinr"]
                if c in df.columns
            ]
            needed = {"timestamp", "ulBytes", "dlBytes", "ulQm", "dlQm", "ulMcs", "dlMcs", "puschSnr", "sinr"}
            if not needed.issubset(set(cols)):
                continue

            p = df.select(cols)
            if "round_id" not in p.columns:
                continue

            p = p.with_columns(
                pl.lit(run_id).cast(pl.Utf8).alias("run_id"),
                pl.lit(str(rnti)).cast(pl.Utf8).alias("rnti"),
                pl.lit(str(bw) if bw is not None else "20 MHz").alias("bandwidth"),
                pl.lit(str(tdd) if tdd is not None else "2-2").alias("tdd"),
                pl.col("round_id").cast(pl.Int64, strict=False).alias("server_round"),
            )

            parts.append(
                p.select([
                    "run_id",
                    "server_round",
                    "timestamp",
                    "rnti",
                    "bandwidth",
                    "tdd",
                    "ulBytes",
                    "dlBytes",
                    "ulQm",
                    "dlQm",
                    "ulMcs",
                    "dlMcs",
                    "puschSnr",
                    "sinr",
                ])
            )

    if not parts:
        raise ValueError(
            f"No PHY rows built from trial folders under {fedavg_root}. "
            "Check trial directory structure and phys_layer files."
        )

    return pl.concat(parts, how="vertical_relaxed")


def derive_fl_percentages(fl_df: pl.DataFrame, fl_round_col: str) -> pl.DataFrame:
    req = [
        "run_id",
        fl_round_col,
        "cid",
        "timestamp",
        "train_time",
        "eval_time",
        "uplink_latency",
        "downlink_latency",
        "start_time",
    ]
    miss = [c for c in req if c not in fl_df.columns]
    if miss:
        raise ValueError(f"FL input missing required columns: {miss}")

    fl = _canonical_run_id_col(fl_df, "run_id").select(req + (["exec_time"] if "exec_time" in fl_df.columns else []))

    cast_cols = ["timestamp", "train_time", "eval_time", "uplink_latency", "downlink_latency", "start_time", fl_round_col]
    if "exec_time" in fl.columns:
        cast_cols.append("exec_time")

    fl = fl.with_columns([pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in cast_cols])
    fl = fl.drop_nulls(["run_id", fl_round_col, "cid", "timestamp"])

    per_cid = (
        fl.group_by(["run_id", fl_round_col, "cid"])
        .agg(
            pl.col("timestamp").min().alias("train_start_s"),
            pl.col("train_time").median().alias("train_s"),
            pl.col("eval_time").median().alias("eval_s"),
            pl.col("uplink_latency").median().alias("uplink_s"),
            pl.col("downlink_latency").median().alias("downlink_s"),
            pl.col("start_time").min().alias("start_time_s"),
        )
        .with_columns(
            pl.col("train_s").fill_null(0.0).clip(lower_bound=0.0),
            pl.col("eval_s").fill_null(0.0).clip(lower_bound=0.0),
            pl.col("uplink_s").fill_null(0.0).clip(lower_bound=0.0),
            pl.col("downlink_s").fill_null(0.0).clip(lower_bound=0.0),
        )
        .with_columns(
            (pl.col("train_start_s") + pl.col("train_s") + pl.col("eval_s")).alias("ul_start_s")
        )
        .with_columns((pl.col("ul_start_s") + pl.col("uplink_s")).alias("ul_end_s"))
    )

    round_level = (
        per_cid.group_by(["run_id", fl_round_col])
        .agg(
            pl.col("train_start_s").min().alias("train_start_min_s"),
            pl.col("ul_end_s").max().alias("uplink_end_s"),
            pl.col("start_time_s").min().alias("start_time_s"),
        )
        .sort(["run_id", fl_round_col])
    )

    if "exec_time" in fl.columns:
        run_exec = fl.group_by("run_id").agg(pl.col("exec_time").max().alias("exec_time_s"))
        round_level = round_level.join(run_exec, on="run_id", how="left")
    else:
        round_level = round_level.with_columns(pl.lit(None, dtype=pl.Float64).alias("exec_time_s"))

    round_level = (
        round_level.with_columns(pl.col("train_start_min_s").alias("round_start_s"))
        .with_columns(pl.col("train_start_min_s").shift(-1).over("run_id").alias("next_start_s"))
        .with_columns((pl.col("start_time_s") + pl.col("exec_time_s")).alias("run_end_s"))
        .with_columns(
            pl.coalesce(["next_start_s", "run_end_s", "uplink_end_s"]).alias("round_end_s")
        )
        .with_columns(
            pl.max_horizontal([pl.col("round_end_s"), pl.col("uplink_end_s")]).alias("round_end_s")
        )
        .with_columns((pl.col("round_end_s") - pl.col("round_start_s")).clip(lower_bound=0.0).alias("round_total_s"))
    )

    per_cid = per_cid.join(
        round_level.select(["run_id", fl_round_col, "round_total_s"]),
        on=["run_id", fl_round_col],
        how="left",
    ).with_columns(
        pl.when(pl.col("round_total_s") > 0)
        .then(pl.col("uplink_s") / pl.col("round_total_s"))
        .otherwise(None)
        .alias("uplink_pct_total"),
        pl.when(pl.col("round_total_s") > 0)
        .then(pl.col("downlink_s") / pl.col("round_total_s"))
        .otherwise(None)
        .alias("downlink_pct_total"),
    )

    min_p = per_cid.group_by(["run_id", fl_round_col]).agg(
        pl.col("uplink_pct_total").min().alias("min_uplink_pct"),
        pl.col("downlink_pct_total").min().alias("min_downlink_pct"),
    )
    return per_cid.join(min_p, on=["run_id", fl_round_col], how="left")


def _extract_cid_from_device_label(v: str) -> Optional[str]:
    s = str(v)
    m = re.search(r"\bue_([0-9]+)_", s)
    if m:
        return m.group(1)
    return None


def resolve_phy_device_to_cid(
    phy_df: pl.DataFrame,
    phy_device_col: str,
    mapping_csv: Optional[Path],
    fl_df: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    keys = _canonical_run_id_col(
        phy_df.select(["run_id", phy_device_col]).unique(),
        "run_id",
    ).with_columns(pl.lit(None, dtype=pl.Int64).alias("cid_mapped"))

    if mapping_csv and mapping_csv.exists():
        mdf = pl.read_csv(mapping_csv, schema_overrides={"run_id": pl.Utf8}).rename({"rnti": "rnti_map", "cid": "cid_map"})
        req = {"run_id", "rnti_map", "cid_map"}
        miss = req - set(mdf.columns)
        if miss:
            raise ValueError(f"mapping csv missing columns: {sorted(miss)}")

        mdf = _canonical_run_id_col(mdf, "run_id").with_columns(
            pl.col("rnti_map").cast(pl.Utf8).str.replace(r"^ue_", "").str.replace(r"\.csv$", "").alias("rnti_map"),
            pl.col("cid_map").cast(pl.Utf8).alias("cid_map"),
        )

        tmp = keys.with_columns(
            pl.col(phy_device_col).cast(pl.Utf8).str.replace(r"^ue_", "").str.replace(r"\.csv$", "").alias("rnti_key")
        ).join(
            mdf.select(["run_id", "rnti_map", "cid_map"]),
            left_on=["run_id", "rnti_key"],
            right_on=["run_id", "rnti_map"],
            how="left",
        )

        # Fallback: if run_id representation mismatches, map by RNTI only.
        rnti_only = (
            mdf.group_by("rnti_map")
            .agg(pl.col("cid_map").mode().first().alias("cid_by_rnti"))
        )
        tmp = tmp.join(rnti_only, left_on="rnti_key", right_on="rnti_map", how="left")
        tmp = tmp.with_columns(
            pl.coalesce(["cid_map", "cid_by_rnti"]).cast(pl.Int64, strict=False).alias("cid_mapped")
        )
        mapped = tmp.select(["run_id", phy_device_col, "cid_mapped"])
    else:
        mapped = keys.with_columns(
        pl.col(phy_device_col)
        .cast(pl.Utf8)
        .map_elements(_extract_cid_from_device_label, return_dtype=pl.Utf8)
        .cast(pl.Int64, strict=False)
        .alias("cid_mapped")
    ).select(["run_id", phy_device_col, "cid_mapped"])

    # Fallback when device labels are not directly parseable and no mapping CSV is provided.
    # Assign unmapped devices to run-local FL CIDs by stable sorted order.
    if fl_df is not None and "cid" in fl_df.columns:
        cid_rows = (
            _canonical_run_id_col(fl_df.select(["run_id", "cid"]), "run_id")
            .with_columns(pl.col("cid").cast(pl.Int64, strict=False).alias("cid"))
            .drop_nulls(["run_id", "cid"])
            .unique()
            .sort(["run_id", "cid"])
        )
        run_to_cids: Dict[str, List[int]] = {}
        for run_id, cid in cid_rows.iter_rows():
            run_to_cids.setdefault(str(run_id), []).append(int(cid))

        out_rows = []
        for g in mapped.partition_by("run_id", as_dict=False, maintain_order=True):
            run_id = str(g.select(pl.col("run_id").first()).item())
            cids = run_to_cids.get(run_id, [])
            if not cids:
                out_rows.append(g)
                continue

            rows = g.sort(phy_device_col).to_dicts()
            next_idx = 0
            for r in rows:
                if r.get("cid_mapped") is None and next_idx < len(cids):
                    r["cid_mapped"] = int(cids[next_idx])
                    next_idx += 1
            out_rows.append(pl.DataFrame(rows))

        if out_rows:
            mapped = pl.concat(out_rows, how="vertical_relaxed")

    return mapped.select(["run_id", phy_device_col, "cid_mapped"])


def top_fraction_mean(values: List[float], frac: float) -> float:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if a.size == 0 or not np.isfinite(frac) or frac <= 0:
        return float("nan")
    f = min(max(float(frac), 0.0), 1.0)
    k = int(np.ceil(a.size * f))
    k = min(max(k, 1), a.size)
    top = np.partition(a, a.size - k)[-k:]
    return float(np.mean(top))


def _top_fraction_window(ts: List, values: List, frac: float) -> Tuple[object, object]:
    pairs = [(t, float(v)) for t, v in zip(ts, values) if t is not None and v is not None and np.isfinite(v)]
    if not pairs or not np.isfinite(frac) or frac <= 0:
        return None, None

    vals = np.asarray([p[1] for p in pairs], dtype=float)
    k = int(np.ceil(len(vals) * min(max(float(frac), 0.0), 1.0)))
    k = min(max(k, 1), len(vals))
    idx = np.argsort(vals)[-k:]
    sel_t = [pairs[i][0] for i in idx]
    return min(sel_t), max(sel_t)


def _range_average_sum_for_metric(
    round_df: pl.DataFrame,
    phy_device_col: str,
    metric_col: str,
    frac_col: str,
) -> float:
    if metric_col not in round_df.columns or frac_col not in round_df.columns or "timestamp" not in round_df.columns:
        return float("nan")

    windows: Dict[str, Tuple[object, object, Dict[object, float]]] = {}
    for dg in round_df.partition_by(phy_device_col, as_dict=False, maintain_order=True):
        dev = str(dg.select(pl.col(phy_device_col).first()).item())
        frac = dg.select(pl.col(frac_col).first()).item()
        frac = float(frac) if frac is not None else float("nan")

        ts = dg.get_column("timestamp").to_list()
        vals = dg.get_column(metric_col).cast(pl.Float64, strict=False).to_list()
        start, end = _top_fraction_window(ts, vals, frac)
        if start is None or end is None:
            continue

        per_ts: Dict[object, List[float]] = {}
        for t, v in zip(ts, vals):
            if t is None or v is None or not np.isfinite(v):
                continue
            per_ts.setdefault(t, []).append(float(v))
        if not per_ts:
            continue
        ser = {t: float(np.mean(vs)) for t, vs in per_ts.items()}
        windows[dev] = (start, end, ser)

    if not windows:
        return float("nan")

    global_start = min(w[0] for w in windows.values())
    global_end = max(w[1] for w in windows.values())

    ts_union = set()
    for _, _, ser in windows.values():
        ts_union.update([t for t in ser.keys() if global_start <= t <= global_end])
    if not ts_union:
        return float("nan")

    sums: List[float] = []
    for t in sorted(ts_union):
        s = 0.0
        active = False
        for start, end, ser in windows.values():
            if start <= t <= end:
                active = True
                if t in ser:
                    s += ser[t]
        if active:
            sums.append(s)

    if not sums:
        return float("nan")
    return float(np.mean(np.asarray(sums, dtype=float)))


def summarize_phy_per_device_round(
    phy_df: pl.DataFrame,
    phy_device_col: str,
    phy_round_col: str,
    fl_pcts: pl.DataFrame,
    fl_round_col: str,
    device_to_cid: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    p_lookup = fl_pcts.select([
        "run_id",
        fl_round_col,
        "cid",
        "uplink_pct_total",
        "downlink_pct_total",
        "min_uplink_pct",
        "min_downlink_pct",
    ]).rename({fl_round_col: phy_round_col, "cid": "cid_mapped"}).with_columns(
        pl.col(phy_round_col).cast(pl.Int64, strict=False).alias(phy_round_col),
        pl.col("cid_mapped").cast(pl.Int64, strict=False).alias("cid_mapped"),
    )

    base = _canonical_run_id_col(phy_df, "run_id").with_columns(
        pl.col(phy_round_col).cast(pl.Int64, strict=False).alias(phy_round_col)
    ).join(device_to_cid, on=["run_id", phy_device_col], how="left")
    base = base.with_columns(pl.col("cid_mapped").cast(pl.Int64, strict=False).alias("cid_mapped"))
    base = base.join(p_lookup, on=["run_id", phy_round_col, "cid_mapped"], how="left")

    min_p = fl_pcts.select(["run_id", fl_round_col, "min_uplink_pct", "min_downlink_pct"]).unique().rename({
        fl_round_col: phy_round_col,
        "min_uplink_pct": "min_uplink_pct_global",
        "min_downlink_pct": "min_downlink_pct_global",
    }).with_columns(pl.col(phy_round_col).cast(pl.Int64, strict=False).alias(phy_round_col))
    base = base.join(min_p, on=["run_id", phy_round_col], how="left")

    base = base.with_columns(
        pl.when(pl.col("cid_mapped").is_not_null())
        .then(pl.col("uplink_pct_total"))
        .otherwise(pl.col("min_uplink_pct_global"))
        .alias("ul_fraction_used"),
        pl.when(pl.col("cid_mapped").is_not_null())
        .then(pl.col("downlink_pct_total"))
        .otherwise(pl.col("min_downlink_pct_global"))
        .alias("dl_fraction_used"),
    )

    out_rows = []
    for g in base.partition_by(["run_id", phy_round_col, phy_device_col], as_dict=False, maintain_order=True):
        run = g.select(pl.col("run_id").first()).item()
        rnd = g.select(pl.col(phy_round_col).first()).item()
        dev = g.select(pl.col(phy_device_col).first()).item()
        cid_mapped = g.select(pl.col("cid_mapped").first()).item()
        ulf = g.select(pl.col("ul_fraction_used").first()).item()
        dlf = g.select(pl.col("dl_fraction_used").first()).item()
        ulf = float(ulf) if ulf is not None else float("nan")
        dlf = float(dlf) if dlf is not None else float("nan")

        row = {
            "run_id": run,
            str(phy_round_col): rnd,
            str(phy_device_col): dev,
            "cid_mapped": cid_mapped,
            "ul_fraction_used": ulf,
            "dl_fraction_used": dlf,
        }
        for m in UL_METRICS:
            row[m] = top_fraction_mean(g.get_column(m).cast(pl.Float64, strict=False).to_list(), ulf)
        for m in DL_METRICS:
            row[m] = top_fraction_mean(g.get_column(m).cast(pl.Float64, strict=False).to_list(), dlf)
        out_rows.append(row)

    round_total_rows = []
    # for directory in tqdm(directories, desc='Processing experiments.'):
    for rg in tqdm(base.partition_by(["run_id", phy_round_col], as_dict=False, maintain_order=True), desc='processing summarize phy per device round'):
        run = rg.select(pl.col("run_id").first()).item()
        rnd = rg.select(pl.col(phy_round_col).first()).item()
        round_total_rows.append(
            {
                "run_id": run,
                str(phy_round_col): rnd,
                "ul_total_throughput_mbps": _range_average_sum_for_metric(
                    rg,
                    phy_device_col=phy_device_col,
                    metric_col="ul_throughput_mbps",
                    frac_col="ul_fraction_used",
                ),
                "dl_total_throughput_mbps": _range_average_sum_for_metric(
                    rg,
                    phy_device_col=phy_device_col,
                    metric_col="dl_throughput_mbps",
                    frac_col="dl_fraction_used",
                ),
            }
        )

    return pl.DataFrame(out_rows), pl.DataFrame(round_total_rows)


def write_outputs(
    fl_df: pl.DataFrame,
    agg_df: pl.DataFrame,
    per_device: pl.DataFrame,
    round_totals: pl.DataFrame,
    fl_round_col: str,
    agg_round_col: str,
    fl_output: Path,
    agg_output: Path,
    per_device_output: Path,
) -> None:
    known = per_device.filter(pl.col("cid_mapped").is_not_null()).with_columns(
        pl.col("cid_mapped").cast(pl.Int64, strict=False).alias("cid")
    )

    per_round_col = None
    for c in [agg_round_col, fl_round_col, "server_round", "round", "round_id"]:
        if c in known.columns:
            per_round_col = c
            break
    if per_round_col is None:
        raise ValueError("Could not identify round column in per-device summaries.")

    known = known.rename({per_round_col: fl_round_col})
    known = known.with_columns(
        pl.col(fl_round_col).cast(pl.Int64, strict=False).alias(fl_round_col),
        pl.col("cid").cast(pl.Int64, strict=False).alias("cid"),
    )

    cid_round = known.group_by(["run_id", fl_round_col, "cid"]).agg([pl.col(m).mean().alias(m) for m in ALL_METRICS])

    fl_out = fl_df.drop([c for c in ALL_METRICS if c in fl_df.columns]).with_columns(
        pl.col(fl_round_col).cast(pl.Int64, strict=False).alias(fl_round_col),
        pl.col("cid").cast(pl.Int64, strict=False).alias("cid"),
    )
    fl_out = fl_out.join(cid_round, on=["run_id", fl_round_col, "cid"], how="left")

    round_avg = per_device.with_columns(
        pl.col(per_round_col).cast(pl.Int64, strict=False).alias(per_round_col)
    ).group_by(["run_id", per_round_col]).agg([pl.col(m).mean().alias(m) for m in ALL_METRICS])
    rt = round_totals.rename({agg_round_col if agg_round_col in round_totals.columns else per_round_col: per_round_col})
    rt = rt.with_columns(pl.col(per_round_col).cast(pl.Int64, strict=False).alias(per_round_col))
    round_avg = round_avg.join(rt, on=["run_id", per_round_col], how="left").rename({per_round_col: agg_round_col})

    agg_out = agg_df.drop([c for c in ALL_METRICS if c in agg_df.columns]).with_columns(
        pl.col(agg_round_col).cast(pl.Int64, strict=False).alias(agg_round_col)
    )
    agg_out = agg_out.join(round_avg, on=["run_id", agg_round_col], how="left")

    per_device_output.parent.mkdir(parents=True, exist_ok=True)
    fl_out.write_csv(fl_output)
    agg_out.write_csv(agg_output)
    per_device.write_csv(per_device_output)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Enrich FL rows in all_data.csv and write round averages to csv.")
    p.add_argument("--fl-input", type=Path, default=Path("/Users/kmcomer/fed5g_analysis/data/all_data.csv"))
    p.add_argument("--phy-input", type=Path, default=Path("/Users/kmcomer/fed5g_analysis/data/phy_built_all_trials.csv"))
    p.add_argument("--agg-input", type=Path, default=Path("/Users/kmcomer/fed5g_analysis/data/all_data_agg.csv"))
    p.add_argument("--ref-csv", type=Path, default=Path("/Users/kmcomer/fed_5g/ref.csv"))
    p.add_argument("--mapping-csv", type=Path, default=None, help="Optional run_id,rnti,cid mapping CSV")
    p.add_argument(
        "--phy-root-dir",
        type=Path,
        default='/Users/kmcomer/Documents/5G Experiment Data/FedAvg',
        help="Optional FedAvg root directory used to build PHY input from per-trial phys_layer files when --phy-input is missing.",
    )
    p.add_argument(
        "--phy-built-output",
        type=Path,
        default="/Users/kmcomer/fed5g_analysis/data/phy_all.csv",
        help="Optional path to save the built PHY input CSV when using --phy-root-dir.",
    )

    p.add_argument("--fl-output", type=Path, default=Path("/Users/kmcomer/fed5g_analysis/data/theo_throughput_mapped_cids_only.csv"))
    p.add_argument("--agg-output", type=Path, default=Path("/Users/kmcomer/fed5g_analysis/data/theo_throughput_agg.csv"))
    p.add_argument("--per-device-output", type=Path, default=Path("/Users/kmcomer/fed5g_analysis/data/theo_throughput.csv"))

    p.add_argument("--fl-round-col", default=None, help="Optional FL round column override")
    p.add_argument("--phy-round-col", default=None, help="Optional PHY round column override")
    p.add_argument("--phy-device-col", default=None, help="Optional PHY device column override")
    return p


def main() -> None:
    args = build_parser().parse_args()
    required_inputs = [args.fl_input, args.agg_input, args.ref_csv]
    for fp in required_inputs:
        if fp is None or not Path(fp).exists():
            raise FileNotFoundError(f"Missing required input: {fp}")

    build_from_root = args.phy_root_dir is not None
    phy_input_exists = args.phy_input is not None and Path(args.phy_input).exists()
    if not phy_input_exists and not build_from_root:
        raise FileNotFoundError(
            "Missing PHY input. Provide --phy-input with an existing CSV, or use --phy-root-dir to build it from trial folders."
        )

    fl_df = _canonical_run_id_col(pl.read_csv(args.fl_input, schema_overrides={"run_id": pl.Utf8}), "run_id")
    agg_df = _canonical_run_id_col(pl.read_csv(args.agg_input, schema_overrides={"run_id": pl.Utf8}), "run_id")

    if build_from_root:
        phy_df = build_phy_input_from_trials(fl_df, args.phy_root_dir)
        phy_df = _canonical_run_id_col(phy_df, "run_id")
        if args.phy_built_output is not None:
            args.phy_built_output.parent.mkdir(parents=True, exist_ok=True)
            phy_df.write_csv(args.phy_built_output)
            print(f"Built PHY input CSV: {args.phy_built_output}")
    else:
        phy_df = _canonical_run_id_col(
            pl.read_csv(
                args.phy_input,
                schema_overrides=PHY_SCHEMA_OVERRIDES,
                infer_schema_length=100_000,
            ),
            "run_id",
        )

    fl_round_col = args.fl_round_col or _pick_col(fl_df, ["round", "server_round", "round_id"], "FL round column")
    phy_round_col = args.phy_round_col or _pick_col(phy_df, ["server_round", "round", "round_id"], "PHY round column")
    phy_device_col = args.phy_device_col or _pick_col(phy_df, ["rnti", "device", "ue", "ue_label", "id"], "PHY device column")

    tmp = phy_df.select(["run_id", phy_round_col, phy_device_col]).drop_nulls()
    nunique = tmp.group_by(["run_id", phy_round_col]).agg(pl.col(phy_device_col).n_unique().alias("n"))
    mean_n = nunique.select(pl.col("n").mean()).to_series().item() if nunique.height > 0 else None
    if nunique.height == 0 or mean_n is None or float(mean_n) <= 1.0:
        raise ValueError(
            "PHY input does not appear to contain per-device measurements (mean unique devices per run/round <= 1). "
            "Provide a PHY table with an RNTI/device key via --phy-input/--phy-device-col."
        )

    phy_enriched = add_throughput_columns_phy(phy_df, args.ref_csv, phy_device_col=phy_device_col)
    fl_pcts = derive_fl_percentages(fl_df, fl_round_col=fl_round_col)
    d2c = resolve_phy_device_to_cid(
        phy_enriched,
        phy_device_col=phy_device_col,
        mapping_csv=args.mapping_csv,
        fl_df=fl_df,
    )

    per_device, round_totals = summarize_phy_per_device_round(
        phy_df=phy_enriched,
        phy_device_col=phy_device_col,
        phy_round_col=phy_round_col,
        fl_pcts=fl_pcts,
        fl_round_col=fl_round_col,
        device_to_cid=d2c,
    )

    write_outputs(
        fl_df=fl_df,
        agg_df=agg_df,
        per_device=per_device,
        round_totals=round_totals,
        fl_round_col=fl_round_col,
        agg_round_col=phy_round_col,
        fl_output=args.fl_output,
        agg_output=args.agg_output,
        per_device_output=args.per_device_output,
    )

    known = per_device.select(pl.col("cid_mapped").is_not_null().sum()).to_series().item()
    total = per_device.height
    print(f"PHY per-device summaries: {total}, mapped-to-CID: {int(known or 0)}")
    print(f"Wrote FL output: {args.fl_output}")
    print(f"Wrote AGG output: {args.agg_output}")
    print(f"Wrote per-device output: {args.per_device_output}")


if __name__ == "__main__":
    main()
