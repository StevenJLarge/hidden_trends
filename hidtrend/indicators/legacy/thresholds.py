# Code for running threshold-based trend-labelling algorithms (legacy)
import pandas as pd
import numpy as np
from typing import Optional

from hidtrend.indicators.legacy.config_legacy import THRESHOLD_METHODS


class BarrierSpec:
    def __init__(
        self, upper_barrier: pd.Series, lower_barrier: pd.Series,
        log_scale: bool
    ):
        self._upper = upper_barrier
        self._lower = lower_barrier
        self._logscale = log_scale

    def __repr__(self):
        return (
            f"BarrierSpec(upper={self._upper}, lower={self._lower}, "
            f"log={self._logscale})"
        )

    @property
    def upper(self):
        return self._upper

    @property
    def lower(self):
        return self._lower


class ThresholdLabelResult:
    def __init__(self, df_result: pd.DataFrame, barriers: BarrierSpec):
        self._results = df_result.dropna()
        self._barriers = barriers

    def __repr__(self):
        return (
            f"ThresholdLabelResult(result_cols={self._results.columns}, "
            f"barriers={self._barriers})"
        )

    @property
    def indicator(self):
        return self._results["indicator"]

    @property
    def up_trend(self):
        return self._results["up_trend"]

    @property
    def down_trend(self):
        return self._results["down_trend"]


def _infer_price_barriers(
    df_px: pd.DataFrame, centered: bool, delta_t: int, log_scaling: bool,
    n_std: Optional[int] = 1.5, window_size: Optional[int] = 63
) -> BarrierSpec:
    # Code for inferring upper and lower price barriers -- This should be
    # theoretically derived
    if log_scaling:
        vol = df_px["price"].diff().rolling(window_size, center=centered).std()
        upper_barrier = df_px["price"] + n_std * np.sqrt(delta_t) * vol
        lower_barrier = df_px["price"] - n_std * np.sqrt(delta_t) * vol

    else:
        vol = (
            df_px["price"]
            .pct_change()
            .rolling(window_size, center=centered)
            .std()
        )
        upper_barrier = df_px["price"] * (1 + n_std * np.sqrt(delta_t) * vol)
        lower_barrier = df_px["price"] * (1 - n_std * np.sqrt(delta_t) * vol)

    return BarrierSpec(upper_barrier, lower_barrier, log_scaling)


def _validate_input(
    barriers: BarrierSpec, infer_barriers: bool, df_px: pd.Series,
    price_col: str
):
    if barriers is None and not infer_barriers:
        raise ValueError(
            "Must provide barriers in `barrier` argument, or set "
            "`infer_barriers` to True"
        )

    if isinstance(df_px, pd.Series):
        df_px = df_px.to_frame()
        df_px.columns = [price_col]

    if price_col not in pd.DataFrame(df_px).columns:
        raise ValueError(f'`price_col` {price_col} not found on input data')

    return df_px


def _preprocess_input(df_px: pd.Series, price_col: str, log_scaling: bool):
    if log_scaling:
        df_proc = pd.DataFrame(df_px[price_col].copy().apply(np.log, raw=True))
        df_proc.columns = ["price"]
    else:
        df_proc = pd.DataFrame(df_px[price_col].copy())
        df_proc.columns = ["price"]
    return df_proc


def _fixed_time_triple_barrier(
    df_px: pd.DataFrame, delta_t: int, infer_barriers: Optional[bool] = False,
    barriers: BarrierSpec = None, centered: bool = True,
    price_col: str = "price", log_scaling: bool = False, n_sigma: float = 1.0
) -> ThresholdLabelResult:
    # Main routine for fixed-time triple-barrier
    df_px = _validate_input(barriers, infer_barriers, df_px, price_col)
    df_res = _preprocess_input(df_px, price_col, log_scaling)

    if infer_barriers:
        barriers = _infer_price_barriers(
            df_res, centered, delta_t, log_scaling, n_std=n_sigma
        )

    df_res["price_fwd"] = df_res["price"].shift(-delta_t)
    df_res["up_trend"] = (df_res["price_fwd"] > barriers.upper).astype(int)
    df_res["down_trend"] = (df_res["price_fwd"] < barriers.lower).astype(int)
    df_res["indicator"] = df_res["up_trend"] - df_res["down_trend"]

    return ThresholdLabelResult(df_res, barriers)


def _first_passage_triple_barrier(
    df_px: pd.DataFrame, delta_t: int, infer_barriers: bool = False,
    barriers: BarrierSpec = None, centered: bool = True,
    price_col: str = "price", log_scaling: bool = False, n_sigma: float = 1.0
) -> ThresholdLabelResult:
    # Main routine for first-passage triple-barrier method
    df_px = _validate_input(barriers, infer_barriers, df_px, price_col)
    df_res = _preprocess_input(df_px, price_col, log_scaling)

    if infer_barriers:
        barriers = _infer_price_barriers(
            df_res, centered, delta_t, log_scaling, n_std=n_sigma
        )

    # Get max value and index of that value in the rolling window
    df_res["px_max"] = (
        df_res["price"]
        .shift(-delta_t)
        .rolling(delta_t)
        .apply(np.max, raw=True)
    )
    df_res["idx_max"] = (
        df_res["price"]
        .shift(-delta_t)
        .rolling(delta_t)
        .apply(np.argmax, raw=True)
    )

    # Same for the minimum
    df_res["px_min"] = (
        df_res["price"]
        .shift(-delta_t)
        .rolling(delta_t)
        .apply(np.min, raw=True)
    )
    df_res["idx_min"] = (
        df_res["price"]
        .shift(-delta_t)
        .rolling(delta_t)
        .apply(np.argmin, raw=True)
    )

    # Classify each time as up, down, or no-trend.
    df_res["trend_up"] = threshold_up_trend(df_res, barriers)
    df_res["trend_down"] = threshold_down_trend(df_res, barriers)
    df_res["indicator"] = df_res["trend_up"] - df_res["trend_down"]

    return ThresholdLabelResult(df_res, barriers)


def threshold_up_trend(df: pd.DataFrame, b: BarrierSpec):
    return (
        ((df["px_max"] > b.upper) & (df["px_min"] > b.lower))
        | (
            (df["idx_max"] < df['idx_min'])
            & (df["px_min"] < b.lower)
            & (df["px_max"] > b.upper)
        )
    ).astype(int)


def threshold_down_trend(df: pd.DataFrame, b: BarrierSpec):
    return (
        ((df["px_min"] < b.lower) & (df['px_max'] > b.upper))
        | (
            (df["idx_min"] < df["idx_max"])
            & (df["px_max"] < b.upper)
            & (df["px_min"] < b.lower)
        )
    ).astype(int)


def triple_barrier(
    df_px: pd.DataFrame, delta_t: int = 63, infer_barriers: bool = True,
    method: str = 'fixed_time', log_scaling: bool = False, n_sigma: float = 1.0
) -> ThresholdLabelResult:
    if method not in THRESHOLD_METHODS:
        raise NotImplementedError(
            f'Input threshold method {method} is not currently implemented, '
            f'must be one of {THRESHOLD_METHODS}'
        )

    if method == 'fixed_time':
        return _fixed_time_triple_barrier(
            df_px, delta_t, infer_barriers=infer_barriers,
            log_scaling=log_scaling, n_sigma=n_sigma
        )

    elif method == 'first_passage':
        return _first_passage_triple_barrier(
            df_px, delta_t, infer_barriers=infer_barriers,
            log_scaling=log_scaling, n_sigma=n_sigma
        )

    raise ValueError(
        "Broken logic in `triple_barrier`, check `THRESHOLD_METHODS` variable "
        "to ensure correct implementations"
    )


if __name__ == "__main__":
    from vipco import bbgclient
    from pathlib import Path

    proj_dir = Path(__file__).parents[2]
    data_dir = proj_dir / "data" / "testing"
    res_dir = proj_dir / "results" / "testing"

    if not Path(data_dir / "sample_data.csv").exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        df_sample = bbgclient.history(
            "SPX Index", "PX_LAST", start_date="1990-01-01"
        )
        df_sample = pd.DataFrame(df_sample)
        df_sample.columns = ["price"]
        df_sample.to_csv(data_dir / "sample_data.csv")

    df_sample = pd.read_csv(
        data_dir / "sample_data.csv", index_col=0, parse_dates=True
    )

    result_fixed = triple_barrier(df_sample, method="fixed_time")
    result_first = triple_barrier(df_sample, method="first_passage")

    if not Path(res_dir).exists():
        Path(res_dir).mkdir(parents=True, exist_ok=True)

    result_fixed._results.to_csv(res_dir / "test_tb_fixed.csv")
    result_first._results.to_csv(res_dir / "test_tb_first.csv")

    print("\n\n\t\t--DONE--\n\n")
