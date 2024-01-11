from typing import Optional
from pathlib import Path
import os
import pandas as pd
import yfinance as yf

from hidtrend.data.generators.config import THESIS_DATA_YF
from hidtrend.data.generators.base import BaseGenerator
from hidtrend.types import Datelike, Pathlike

_DATA_FIELD = 'Adj Close'


class YahooDataGenerator(BaseGenerator):
    def __init__(
        self, loadfile: Optional[Pathlike] = None,
        savefile: Optional[Pathlike] = None,
        start_date: Optional[Datelike] = None
    ):
        super().__init__(loadfile, savefile)
        self.start_date = start_date

    def generate_data(self, force: Optional[bool] = False):
        if self.loadfile is None or not os.path.exists(self.loadfile) or force:
            self.data = self._fetch_yahoo_data()

        if self.savefile is not None:
            Path(self.savefile.parents[0]).mkdir(parents=True, exist_ok=True)
            self.data.to_parquet(self.savefile)

    def _fetch_yahoo_data(self):
        tickers = list(THESIS_DATA_YF.keys())
        column_map = {k: v['name'] for k, v in THESIS_DATA_YF.items()}

        df = yf.download(tickers)
        df = (
            df.loc[:, pd.IndexSlice[_DATA_FIELD, :]]
            .droplevel(axis=1, level=0)
        )
        df = df.rename(columns=column_map)

        # Start all at 1
        return df.pct_change().add(1).cumprod()


if __name__ == "__main__":
    proj_dir = Path(__file__).resolve().parents[3]
    data_dir = proj_dir / "data" / "thesis"
    gen = YahooDataGenerator(savefile=data_dir / "raw_data.pq")
    df = gen.get_data()

    print('DONE')

