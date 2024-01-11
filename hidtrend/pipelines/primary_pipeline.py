from typing import Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from datetime import datetime
import itertools
import pickle
import numpy as np
import pandas as pd

import hidden_py as hp

from hidtrend.types import Pathlike
from hidtrend.data.generators.base import BaseGenerator
from hidtrend.data import data_thesis
from hidtrend.indicators.legacy import thresholds


proj_dir = Path(__file__).resolve().parents[2]
CACHE_DIR = proj_dir / "data" / "thesis" / "cache"


class HMMDatasetResult:
    def __init__(self, res_dict: dict):
        self._results = res_dict

    def write(self, write_file: Path):
        Path(write_file.parents[0]).mkdir(parents=True, exist_ok=True)
        with open(write_file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read(read_name: Path):
        with open(read_name, 'rb') as f:
            res = pickle.load(f)
        return res


class ThresholdDatasetResult:
    def __init__(self, res_dict: dict):
        self._results = res_dict

    def result_set(self, delta_t: int, n_sigma: str, method: str):
        return (
            self._results[n_sigma][str(delta_t)]
            .loc[:, pd.IndexSlice[:, method]]
            .droplevel(axis=1, level=1)
        )

    @property
    def n_sig_vals(self):
        return list(self._results.keys())

    @property
    def dt_vals(self):
        n_sig = list(self._results.keys())[0]
        return list(self._results[n_sig].keys())

    def write(self, write_file: Path):
        Path(write_file.parents[0]).mkdir(parents=True, exist_ok=True)
        with open(write_file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def read(read_name: Path):
        with open(read_name, 'rb') as f:
            res = pickle.load(f)
        return res


def fetch_market_data(
    savefile: Optional[Pathlike] = None, loadfile: Optional[Pathlike] = None,
    force: bool = False
):
    data_gen = data_thesis.YahooDataGenerator(
        loadfile=loadfile, savefile=savefile
    )
    _ = data_gen.get_data(force=force)
    return data_gen


def _get_trend_indicator_results_threshold(
    dataset: pd.DataFrame, delta_t: int, n_sigma: float = 1.0
):

    res_ft = thresholds.triple_barrier(
        dataset, delta_t, infer_barriers=True, method='fixed_time',
        log_scaling=True, n_sigma=n_sigma
    )

    res_fp = thresholds.triple_barrier(
        dataset, delta_t, infer_barriers=True, method='first_passage',
        log_scaling=True, n_sigma=n_sigma
    )

    return pd.concat(
        [res_ft.indicator, res_fp.indicator],
        keys=['fixed_time', 'first_passage'], axis=1
    )


def calculate_trend_threshold_indicators(
    data_gen: BaseGenerator, delta_t: int = 21, n_sigma: float = 1.0
):
    assets = data_gen.data.columns
    res_dict = {}

    for asset in assets:
        res_dict[asset] = _get_trend_indicator_results_threshold(
            data_gen.data[asset], delta_t, n_sigma=n_sigma,
        )
    return pd.concat(res_dict.values(), keys=res_dict.keys(), axis=1)


def run_indicator_analysis(market_data: pd.DataFrame):
    delta_t_vals = [5, 10, 15, 21, 42, 63]
    sigma_vals = {'0.5': 0.5, '1.0': 1.0, '1.5': 1.5, '2.0': 2.0}

    res_dict = {}

    for sigma_key, sigma_val in sigma_vals.items():
        res_dict[sigma_key] = {}
        for dt in delta_t_vals:
            ind_data = calculate_trend_threshold_indicators(
                market_data, delta_t=dt, n_sigma=sigma_val
            )
            res_dict[sigma_key][str(dt)] = ind_data

    return ThresholdDatasetResult(res_dict)


def get_empirical_trans_init(obs_ts, min_val: float = 0.001):

    n_state_obs = [
        np.max([np.sum(obs_ts == 0), 1]),
        np.max([np.sum(obs_ts == 1), 1]),
        np.max([np.sum(obs_ts == 2), 1])
    ]

    emp_mat = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            mask = np.logical_and(obs_ts[:-1] == j, obs_ts[1:] == i)
            freq = np.sum(mask) / n_state_obs[j]
            emp_mat[i, j] = np.max([freq, min_val])

    emp_mat /= emp_mat.sum(axis=0)
    return emp_mat


def run_hmm_resampling(
    dataset: pd.Series, block_size: int, stride: int, mp: bool = True
):

    _ds = (dataset.ffill().dropna() + 1).astype(int).to_numpy()

    trans_init = get_empirical_trans_init(_ds)
    obs_init = np.array([
        [0.92, 0.05, 0.03],
        [0.05, 0.90, 0.05],
        [0.03, 0.05, 0.92]
    ])

    n_blocks = (_ds.shape[0] - block_size) // stride

    data_blocks = [
        _ds[idx * stride: (idx * stride) + block_size]
        for idx in range(n_blocks)
    ]

    if mp:
        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            _res = executor.map(
                _hmm_block_optimization, data_blocks,
                itertools.repeat(trans_init), itertools.repeat(obs_init)
            )
    else:
        _res = [
            _hmm_block_optimization(blk, trans_init, obs_init)
            for blk in data_blocks
        ]

    return list(_res)


def _hmm_block_optimization(
    data_block: np.ndarray, trans_init: np.ndarray, obs_init: np.ndarray
):
    analyzer = hp.infer.MarkovInfer(3, 3)
    opt_res = analyzer.optimize(
        data_block, trans_init, obs_init, opt_type=hp.OptClass.ExpMax
    )
    return opt_res


def run_hmm_analysis(
    dataset: pd.DataFrame, block_size: int = 252, stride: int = 126,
):
    res_dict = {}
    for c in dataset.columns:
        start_time = datetime.now()
        logger.info(f"Running analysis for {c}...")
        res_dict[c] = run_hmm_resampling(dataset[c], block_size, stride)
        end_time = datetime.now()
        logger.info(f"Exec time -- {str(end_time - start_time)}")

    return res_dict


def run_hmm_fitting(ind_res: ThresholdDatasetResult):
    res_dict = {}

    for n_sig in ind_res.n_sig_vals:
        logger.info(f'Starting HMM calculations for n_sig = {n_sig}')
        res_dict[n_sig] = {}
        for dt in ind_res.dt_vals:
            res_dict[n_sig][dt] = {}

            hmm_results_ft = (
                run_hmm_analysis(
                    ind_res.result_set(dt, n_sig, 'fixed_time')
                )
            )
            hmm_results_fp = (
                run_hmm_analysis(
                    ind_res.result_set(dt, n_sig, 'first_passage')
                )
            )
            res_dict[n_sig][dt]['fixed_time'] = hmm_results_ft
            res_dict[n_sig][dt]["first_passage"] = hmm_results_fp

        logger.info(f"Dumping intermediate result for n_sigma = {n_sig}...")
        _dump_res(res_dict[n_sig], n_sig)

    return HMMDatasetResult(res_dict)


def _dump_res(res_dict: dict, n_sigma: float):
    with open(CACHE_DIR / f'inter_res_hmm_ns{n_sigma}.pkl', 'wb') as f:
        pickle.dump(res_dict, f)


if __name__ == '__main__':
    from loguru import logger

    proj_dir = Path(__file__).resolve().parents[2]
    load_file = proj_dir / "data" / 'thesis' / 'raw_data.pq'

    # Substep outputs
    threshold_output_file = (
        proj_dir / "results" / 'thesis' / 'threshold_res.pkl'
    )
    hmm_output_file = proj_dir / "results" / "thesis" / "hmm_res.pkl"

    # Build data generator and populate dataset
    logger.info('Fetching market data')
    data_gen = fetch_market_data(loadfile=load_file, force=True)

    # run triple-barrier indicator analysis
    logger.info('Calculating indicator data')
    indicator_result = run_indicator_analysis(data_gen)
    indicator_result.write(threshold_output_file)
    indicator_result = ThresholdDatasetResult.read(threshold_output_file)

    # run hmm resampling analysis
    logger.info('Calculating hmm results')
    hmm_results = run_hmm_fitting(indicator_result)
    hmm_results.write(hmm_output_file)
    # hmm_results = HMMDatasetResult.read(hmm_output_file)

    print('-- DONE -- ')
