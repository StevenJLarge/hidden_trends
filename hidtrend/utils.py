from typing import Iterable
from functools import reduce
import pandas as pd


def average_dfs(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return reduce(lambda x, y: x.add(y, fill_value=0), dfs) / len(dfs)
