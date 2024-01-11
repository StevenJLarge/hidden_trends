from pathlib import Path
from typing import Union
from datetime import datetime
import pandas as pd

Pathlike = Union[str, Path]
Datelike = Union[str, datetime, pd.Timestamp]
