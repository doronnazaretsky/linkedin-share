import pandas as pd
import numpy as np


def best_scalar_dtype(series, signed_default=True):
    max_val = series.max()
    if series.min() < 0 or signed_default:
        signed_dtype = True
    else:
        signed_dtype = False
    if signed_dtype:
        max_val = abs(max_val) * (-1)
    best_dtype = np.min_scalar_type(max_val)
    return best_dtype


def downcast_scalar_dtypes(org_df, signed_default=True):
    df = pd.DataFrame(org_df)
    _start = df.memory_usage(deep=True).sum() / 1024 ** 2
    for column in df:
        if df[column].dtype.kind in "i,f":
            best_dtype = best_scalar_dtype(df[column], signed_default)
            df[column] = df[column].astype(best_dtype)
    _end = df.memory_usage(deep=True).sum() / 1024 ** 2
    saved = (_start - _end) / _start * 100
    print(f"Saved {saved:.2f}%")
    return df
