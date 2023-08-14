import torch
from model import data_layer
import datetime
import torch.utils.data as torch_data
import transformers
import sklearn.metrics as metrics


def TrainOn(dataset: torch_data.TensorDataset,
            epochs: int,
            debug: bool = False) -> str:
  ''' Train a transformer classifier and returns the trained model path. '''
  # Load a pre-trained BERT model for sequence classification (binary)
  model = transformers.BertForSequenceClassification.from_pretrained(
      'bert-base-uncased', num_labels=2)
  # Fine-tune using Adam, with Weight Deday (L2 regularization) to prevent overfitting.
  optimizer = transformers.AdamW(model.parameters(), lr=1e-5)

  training_dataset, val_dataset = data_layer.SplitDataset(dataset)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(f'Training model on {device.type}')
  model.to(device)

  if debug:
    print(f'Training model with DEBUG mode')
  for epoch in range(1 if debug else epochs):
    model.train()
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

      if debug:
        break  # Train a single batch for debug.

    # Validation
    model.eval()
    actual, prediction = [], []

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
        actual.extend(batch_labels.cpu().numpy())
        prediction.extend(predicted_labels.cpu().numpy())
    print(f"Epoch {epoch + 1}/{epochs}")
    print(metrics.classification_report(y_true=actual, y_pred=prediction))

  model_path = f'./model_{datetime.datetime.now().strftime("%H_%M_%S")}.pth'
  print(f'Saving trained model to {model_path}')
  torch.save(model.state_dict(), model_path)

  return model_path


class NewsBinaryClassifier:

  def __init__(self, model_path: str):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.tokenizer = transformers.BertTokenizer.from_pretrained(
        'bert-base-uncased')
    self.model = transformers.BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2)
    self.model.load_state_dict(torch.load(model_path, map_location=self.device))

  def Predict(self, url: str) -> int:
    ''' Returns the predicted class for the `text`. 1 for is news, 0 for not news. '''
    #TODO: Multi texts input.
    input = self.tokenizer(data_layer.ProcessInput(url=url),
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