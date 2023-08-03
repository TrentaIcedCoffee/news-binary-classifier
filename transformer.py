from data import ReadCsvToDataFrame

import torch
from data import ProcessInput, ProcessDataFrame, FormDataset, SplitDataset
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report


def TrainOn(labeled_data_path: str, epochs: int, debug: bool = False) -> str:
  ''' Train a transformer classifier and returns the trained model path. '''
  # Load a pre-trained BERT model for sequence classification (binary)
  model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                        num_labels=2)
  # Fine-tune using Adam, with Weight Deday (L2 regularization) to prevent overfitting.
  optimizer = AdamW(model.parameters(), lr=1e-5)

  training_dataset, val_dataset = SplitDataset(
      FormDataset(
          ProcessDataFrame(ReadCsvToDataFrame(labeled_data_path, header=0))))

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

  return model_path


class NewsBinaryClassifier:

  def __init__(self, model_path: str):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)
    self.model.load_state_dict(torch.load(model_path, map_location=self.device))

  def Predict(self, url: str, text: str) -> int:
    ''' Returns the predicted class for the `text`. 1 for is news, 0 for not news. '''
    #TODO: Multi texts input.
    input = self.tokenizer(ProcessInput(url=url, text=text),
                           padding=True,
                           truncation=True,
                           return_tensors='pt')

    with torch.no_grad():
      self.model.eval()
      outputs = self.model(input['input_ids'],
                           attention_mask=input['attention_mask'])
      logits = outputs.logits
      probabilities = torch.softmax(logits, dim=1)
      predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class

  def ValidateOn(self, labeled_data_path: str):
    labeled = ReadCsvToDataFrame(labeled_data_path, header=0)
    labeled[['url', 'text']] = labeled[['url', 'text']].fillna("")
    labeled['is_news'] = labeled['is_news'].fillna(0).astype(int)
    labeled['prediction'] = labeled.apply(lambda row: self.Predict(
        ProcessInput(url=row['url'], text=row['text'])),
                                          axis='columns')
    print(classification_report(labeled['is_news'], labeled['prediction']))


if __name__ == "__main__":
  model_path = TrainOn('./labeled.csv', 50, debug=True)
  NewsBinaryClassifier(model_path).ValidateOn('./labeled.csv')