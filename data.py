from typing import Union
import pandas as pd


def ReadCsvToDataFrame(path: str, header: Union[int, None]) -> pd.DataFrame:
  if header != 0 and header is not None:
    raise f'Expect header to be either 0 (first row) or None (now header), had {header}'

  lines = 0
  with open(path, 'r', encoding='utf-8') as file:
    for _ in file:
      lines += 1
  expect_rows = lines if header is None else lines - 1

  df = pd.read_csv(path, header=header, on_bad_lines='warn')

  if expect_rows != df.shape[0]:
    print(
        f'Raw data has {expect_rows} rows from {lines} lines, found errors in {expect_rows-df.shape[0]} rows when loading to data frame'
    )
  return df
