from typing import Optional
from abc import ABC, abstractmethod

import pandas as pd

from hidtrend.types import Pathlike


class BaseGenerator(ABC):
    def __init__(
        self, loadfile: Optional[Pathlike] = None,
        savefile: Optional[Pathlike] = None
    ):
        self.loadfile = loadfile
        self.savefile = savefile
        self.data = None

    def get_data(self, force: Optional[bool] = False) -> pd.DataFrame:
        if force or self.data is None:
            self.generate_data(force)
            return self.data
        else:
            return self.data

    @abstractmethod
    def generate_data(self, force: Optional[bool] = False):
        pass

    def write_data(self, file: Optional[Pathlike] = None) -> None:
        if file is not None:
            f = file
        elif self.savefile is not None:
            f = self.savefile
        else:
            raise ValueError(
                'No place to write data. `file` and `self.file` '
                'cannot both be None.'
            )
        print(f'writing data to {f}')
        self.data.to_parquet(f)

