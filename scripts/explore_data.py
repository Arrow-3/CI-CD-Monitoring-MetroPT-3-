import pandas as pd
from prism.config import settings

df = pd.read_csv(settings.DATA_PATH, nrows=200_000)
print("shape:", df.shape)
print("columns:", list(df.columns))
print(df.head())
print(df.describe().T[["min", "max", "mean", "std"]])
