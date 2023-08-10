from typing import Tuple
import pandas as pd
import transformers
import torch
import torch.utils.data as torch_data


def LoadLabeledData(path: str) -> pd.DataFrame:
  labeled = pd.read_csv(path, header=0, on_bad_lines='warn')

  # Merge labelling results.
  labeled['is_news'] = labeled.apply(
      lambda row: row['is_news (B)']
      if not pd.isna(row['is_news (B)']) else row['is_news (A)'],
      axis='columns')

  # Drop rows with missing values.
  initial_indices = labeled.index
  labeled = labeled.dropna(subset=['url', 'is_news'])
  dropped_indices = initial_indices.difference(labeled.index).to_list()
  if len(dropped_indices) > 0:
    print(
        f'Dropped indices with missing values {initial_indices.difference(labeled.index).to_list()}'
    )

  # Formatting columns.
  labeled['url'] = labeled['url'].astype(str)
  labeled['is_news'] = labeled['is_news'].astype(int)

  print(f'Loaded data of shape {labeled.shape}')

  return labeled


def ProcessInput(url: str) -> str:
  ''' Input mapper. Both training and validation should apply this input mapper on raw data. '''
  return url.lower()


def FormDataset(labeled: pd.DataFrame) -> torch_data.TensorDataset:
  ''' Generates a tensor dataset for training. '''
  # Form inputs and labels for training.
  inputs = labeled.apply(lambda row: ProcessInput(url=row['url']),
                         axis='columns').to_list()
  # Word embedding using a pre-trained BERT tokenizer.
  tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
  encoded_input = tokenizer(inputs,
                            padding=True,
                            truncation=True,
                            return_tensors='pt')

  labels = torch.tensor(labeled['is_news'].to_list())

  # TODO: Peek the dataset.
  dataset = torch_data.TensorDataset(encoded_input['input_ids'],
                                     encoded_input['attention_mask'], labels)

  print(f'Formed a dataset of size {len(dataset)}')

  return dataset


def SplitDataset(
    dataset: torch_data.TensorDataset
) -> Tuple[torch_data.DataLoader, torch_data.DataLoader]:
  split_ratio = 0.8
  training_dataset_size = int(len(dataset) * split_ratio)
  val_dataset_size = len(dataset) - training_dataset_size

  train_dataset, val_dataset = torch_data.random_split(
      dataset, [training_dataset_size, val_dataset_size])

  print(
      f'Splited dataset of size {len(dataset)} into {len(train_dataset)} training and {len(val_dataset)} validation.'
  )

  return torch_data.DataLoader(train_dataset, batch_size=16,
                               shuffle=True), torch_data.DataLoader(
                                   val_dataset, batch_size=16)
