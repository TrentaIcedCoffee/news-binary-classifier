from typing import Union, Tuple
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer
import torch


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


def ProcessInput(url: str, text: str) -> str:
  return url.lower() + ':' + text.lower()


def ProcessDataFrame(labeled: pd.DataFrame) -> pd.DataFrame:
  # Fill missing values.
  labeled[['url', 'text']] = labeled[['url', 'text']].fillna("")
  labeled['is_news'] = labeled['is_news'].fillna(0).astype(int)


def FormDataset(labeled: pd.DataFrame) -> TensorDataset:
  # Form inputs and labels for training.
  inputs = labeled.apply(
      lambda row: ProcessInput(url=row['url'], text=row['text']),
      axis='columns').to_list()
  labels = labeled['is_news'].to_list()

  # Word embedding using a pre-trained BERT tokenizer.
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  encoded_input = tokenizer(inputs,
                            padding=True,
                            truncation=True,
                            return_tensors='pt')
  labels = torch.tensor(labels)

  # TODO: Peek the dataset.
  dataset = TensorDataset(encoded_input['input_ids'],
                          encoded_input['attention_mask'], labels)

  print(f'Processed a dataset of size {len(dataset)}')

  return dataset


def SplitDataset(dataset: TensorDataset) -> Tuple[DataLoader, DataLoader]:
  split_ratio = 0.8
  training_dataset_size = int(len(dataset) * split_ratio)
  val_dataset_size = len(dataset) - training_dataset_size

  train_dataset, val_dataset = random_split(
      dataset, [training_dataset_size, val_dataset_size])

  print(
      f'Splited dataset of size {len(dataset)} into {len(train_dataset)} training and {len(val_dataset)} validation.'
  )

  return DataLoader(train_dataset, batch_size=16,
                    shuffle=True), DataLoader(val_dataset, batch_size=16)
