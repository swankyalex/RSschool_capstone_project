import os

import pandas as pd
from pandas_profiling import ProfileReport

from src.forest_model.consts import DATA_PATH

path = os.path.join(DATA_PATH, "train.csv")
data = pd.read_csv(path)
profile = ProfileReport(data, title="Pandas Profiling Report", minimal=True)
profile.to_file(os.path.join(DATA_PATH, "forest_report.html"))
