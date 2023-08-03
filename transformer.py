from data import ReadCsvToDataFrame

from typing import Tuple
import pandas as pd
import torch
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split


def ProcessDataset(labeled: pd.DataFrame):
  # Fill missing values.
  labeled[['url', 'text']] = labeled[['url', 'text']].fillna("")
  labeled['is_news'] = labeled['is_news'].fillna(0).astype(int)

  # Form inputs and labels for training.
  inputs = (labeled['url'] + ' ' + labeled['text'].str.lower()).to_list()
  labels = labeled['is_news'].to_list()

  # Word embedding using a pre-trained BERT tokenizer.
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  encoded_input = tokenizer(inputs,
                            padding=True,
                            truncation=True,
                            return_tensors='pt')
  labels = torch.tensor(labels)

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


def TrainOn(labeled_data_path: str, epochs: int, debug: bool = False) -> None:
  # Load a pre-trained BERT model for sequence classification (binary)
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                        num_labels=2)
  # Fine-tune using Adam, with Weight Deday (L2 regularization) to prevent overfitting.
  optimizer = AdamW(model.parameters(), lr=1e-5)

  training_dataset, val_dataset = SplitDataset(
      ProcessDataset(ReadCsvToDataFrame(labeled_data_path, header=0)))

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Training model on {device.type}')
  model.to(device)

  if debug:
    print(f'Training model with debug mode')
  for epoch in range(1 if debug else epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(training_dataset):
      print(
          f'Training epoch {epoch+1}/{epochs}, batch {batch_idx+1}/{len(training_dataset)}'
      )
      input_ids, attention_mask, batch_labels = batch
      input_ids, attention_mask, batch_labels = input_ids.to(
          device), attention_mask.to(device), batch_labels.to(device)

      # Forward pass
      outputs = model(input_ids,
                      attention_mask=attention_mask,
                      labels=batch_labels)
      loss = outputs.loss

      # Backward pass and optimization
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      total_loss += loss.item()

      if debug:
        break  # Train a single batch for debug.

    avg_train_loss = total_loss / len(training_dataset)

    # Validation
    model.eval()
    val_accuracy = 0
    val_steps = 0

    with torch.no_grad():
      for batch in val_dataset:
        input_ids, attention_mask, batch_labels = batch
        input_ids, attention_mask, batch_labels = input_ids.to(
            device), attention_mask.to(device), batch_labels.to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Calculate accuracy
        _, predicted_labels = torch.max(logits, 1)
        val_accuracy += (predicted_labels == batch_labels).sum().item()
        val_steps += len(batch_labels)

    avg_val_accuracy = val_accuracy / val_steps

    print(
        f"Epoch {epoch + 1}/{epochs}, Avg. Training Loss: {avg_train_loss}, Validation Accuracy: {avg_val_accuracy}"
    )

  model_path = f'./model_{datetime.now().strftime("%H_%M_%S")}.pth'
  print(f'Saving trained model to {model_path}')
  torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
  TrainOn('./labeled.csv', 50, debug=True)
