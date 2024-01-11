import os
import numpy as np
import pandas as pd
from typing import Optional

# Fitting methods for linear regressions
import statsmodels.api as sm
from arch import arch_model

# from trendlab.types.threshold import PriceData
from hidtrend.indicators.legacy.config_legacy import (
    FITTING_METHODS, ARCH_VOL_MODELS, ARCH_MEAN_MODELS, ARCH_DIST_MODELS
)


class ModellingLabelResult:
    def __init__(self, df_result: pd.DataFrame):
        self._results = df_result.dropna()

    @property
    def beta(self):
        return self._results["beta"]

    @property
    def beta_up(self):
        return self._results[self._results["up_trend"] == 1]["beta"]

    @property
    def beta_down(self):
        return self._results[self._results["down_trend"] == 1]["beta"]

    @property
    def upper_bound(self):
        return self._results["upper_bound"]

    @property
    def lower_bound(self):
        return self._results["lower_bound"]

    @property
    def up_trend(self):
        return self._results["up_trend"]

    @property
    def down_trend(self):
        return self._results["down_trend"]

    @property
    def indicator(self):
        return self._results["indicator"]


def _validate_input(
    df_px: pd.Series, price_col: str
):
    if isinstance(df_px, pd.Series):
        df_px = df_px.to_frame()
        df_px.columns = [price_col]

    if price_col not in pd.DataFrame(df_px).columns:
        raise ValueError(f'`price_col` {price_col} not found on input data')

    return df_px


def _validate_arch_parameters(
    mean_model: str, vol_model: str, dist_model: str
):
    if mean_model not in ARCH_MEAN_MODELS:
        raise ValueError(
            f"Unsupported mean model{mean_model} for arch model, must be one "
            f"of {ARCH_MEAN_MODELS}"
        )

    if vol_model not in ARCH_VOL_MODELS:
        raise ValueError(
            f"Unsupported volatility model{vol_model} for arch model, must be "
            f"one of {ARCH_VOL_MODELS}"
        )

    if dist_model not in ARCH_DIST_MODELS:
        raise ValueError(
            f"Unsupported distribution model{dist_model} for arch model, must "
            f"be one of {ARCH_DIST_MODELS}"
        )


class LinearResult:
    def __init__(
        self, beta: float, lower_bound: float, upper_bound: float,
        up_trend: int, down_trend: int, indicator: int
    ):
        self.beta = beta
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.up_trend = up_trend
        self.down_trend = down_trend
        self.indicator = indicator


def _preprocess_input(df_px: pd.Series, price_col: str, log_scaling: bool):
    if log_scaling:
        df_proc = pd.DataFrame(df_px[price_col].dropna().copy().apply(np.log, raw=True))
        df_proc.columns = ["price"]
    else:
        df_proc = pd.DataFrame(df_px[price_col].dropna().copy())
        df_proc.columns = ["price"]
    return df_proc


def _run_linear_regression(
    px: pd.Series, alpha: Optional[float] = 0.05
) -> pd.Series:
    x = range(px.dropna().shape[0])
    # model = sm.OLS(px.dropna() / px.dropna().iloc[0] - 1, np.array(x))
    model = sm.OLS(px.dropna(), sm.add_constant(np.array(x)))
    res = model.fit()

    beta = res.params[1]
    conf_int = res.conf_int(alpha=alpha)
    lower_bound = conf_int.iloc[0, 0]
    upper_bound = conf_int.iloc[0, 0]

    up_trend = int((beta > 0) & (lower_bound * upper_bound > 0))
    down_trend = int((beta < 0) & (lower_bound * upper_bound > 0))
    indicator = up_trend - down_trend

    return pd.Series(
        [beta, lower_bound, upper_bound, up_trend, down_trend, indicator]
    )


def _run_arima_fit(
    px: pd.Series, mean_model: str, vol_model: str, dist_model: str,
    log_scaling: Optional[bool] = True, alpha: Optional[float] = 0.05
) -> pd.Series:

    if log_scaling:
        rtns = px.diff().dropna()
    else:
        rtns = pd.pct_change().dropna()

    scale_factor = 1
    if rtns.mean().abs().values[0] < 1 and rtns.mean().abs().values[0] > 0:
        oom = np.floor(np.log10(1 / rtns.mean().abs())).values[0]
        if oom < 1: oom += 1
        scale_factor = 10 ** (oom - 1)

    rtns = scale_factor * rtns

    model = arch_model(
        rtns, mean=mean_model, vol=vol_model, dist=dist_model, rescale=False
    )
    result = model.fit(disp='off')

    mu = result._params[0]
    mu_ci_lower, mu_ci_upper = result.conf_int(alpha=alpha).iloc[0].to_list()
    significant = (mu_ci_upper * mu_ci_lower > 0)

    up_trend = int((mu > 0) & significant)
    down_trend = int((mu < 0) & significant)
    indicator = up_trend - down_trend

    return pd.Series(
        [mu / scale_factor, mu_ci_lower / scale_factor,
        mu_ci_upper / scale_factor, up_trend, down_trend, indicator]
    )


def _linear_trend_fit(
    df_px: pd.DataFrame, delta_t: int, centered: Optional[bool] = True,
    price_col: Optional[str] = "price", log_scaling: Optional[bool] = True,
) -> ModellingLabelResult:
    df_px = _validate_input(df_px, price_col)
    df_res = _preprocess_input(df_px, price_col, log_scaling)

    # Rolling cant return multiple values alone, so we need to run this as a'
    # native python comprehension
    fit_res = pd.concat([
            _run_linear_regression(_df)
            for _df in df_res.rolling(delta_t, center=centered)
        ], axis=1
    )

    fit_res = fit_res.T
    fit_res.columns = [
        "beta", "lower_bound", "upper_bound", "up_trend", "down_trend",
        "indicator"
    ]
    fit_res.index = df_res.index

    return ModellingLabelResult(fit_res)


def _arima_trend_fit(
    df_px: pd.DataFrame, delta_t: int, centered: Optional[bool] = True,
    price_col: Optional[str] = "price", log_scaling: Optional[bool] = True,
    mean_model: Optional[str] = "Constant", vol_model: Optional[str] = "GARCH",
    dist_model: Optional[str] = "normal"
) -> ModellingLabelResult:
    df_px = _validate_input(df_px, price_col)
    _validate_arch_parameters(mean_model, vol_model, dist_model)
    df_res = _preprocess_input(df_px, price_col, log_scaling)

    fit_res = pd.concat([
            _run_arima_fit(_df, mean_model, vol_model, dist_model)
            for _df in df_res.rolling(delta_t, center=centered)
        ], axis=1
    )

    fit_res = fit_res.T
    fit_res.columns = [
        "beta", "lower_bound", "upper_bound", "up_trend", "down_trend",
        "indicator"
    ]
    fit_res.index = df_res.index

    return ModellingLabelResult(fit_res)


def trend_fitting(
    df_px: pd.DataFrame, delta_t: Optional[int] = 63,
    method: Optional[str] = 'linear'
) -> ModellingLabelResult:
    if method not in FITTING_METHODS:
        raise NotImplementedError(
            f'Input fitting method {method} is not currently implemented, '
            f'must be one of {FITTING_METHODS}'
        )

    if method == 'linear':
        return _linear_trend_fit(df_px, delta_t)

    elif method == 'arima':
        return _arima_trend_fit(df_px, delta_t)

    raise ValueError(
        "Broken logic in `trend_fitting`, check `FITTING_METHODS` variable "
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
        df_sample = bbgclient.history("SPX Index", "PX_LAST", start_date="1990-01-01")
        df_sample = pd.DataFrame(df_sample)
        df_sample.columns = ["price"]
        df_sample.to_csv(data_dir / "sample_data.csv")

    df_sample = pd.read_csv(data_dir / "sample_data.csv", index_col=0, parse_dates=True)

    # result_linear = trend_fitting(df_sample, method="linear")
    result_arima = trend_fitting(df_sample, method="arima")

    if not Path(res_dir).exists():
        Path(res_dir).mkdir(parents=True, exist_ok=True)

    # result_linear._results.to_csv(res_dir / "test_tm_linear.csv")
    result_arima._results.to_csv(res_dir / "test_tm_arima.csv")

    print("\n\n\t\t--DONE--\n\n")