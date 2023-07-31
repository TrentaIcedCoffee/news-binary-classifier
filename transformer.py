from data import LoadLabeledData

import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

kDebug = True

labeled = LoadLabeledData()
train_texts = labeled['text'].to_list()
train_labels = labeled['is_news'].to_list()

# Word embedding using a pre-trained BERT tokenizer.
# TODO: Cased?
encoded_texts = BertTokenizer.from_pretrained('bert-base-uncased')(
    train_texts, padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(train_labels)

# Split to train and eval datasets.
dataset = TensorDataset(encoded_texts['input_ids'],
                        encoded_texts['attention_mask'], labels)
train_val_split_ratio = 0.7
train_dataset, val_dataset = random_split(dataset, [
    int(train_val_split_ratio * len(dataset)),
    len(dataset) - int(train_val_split_ratio * len(dataset))
])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load pre-trained BERT model for sequence classification (binary)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=2)
# Fine-tune using Adam, with Weight Deday (L2 regularization) to prevent overfitting.
optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
num_epochs = 3
for epoch in range(num_epochs):

  model.train()
  total_loss = 0
  for batch_idx, batch in enumerate(train_loader):
    print(
        f'Training epoch {epoch+1}/{num_epochs}, batch {batch_idx+1}/{len(train_loader)}'
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
    
    if kDebug:
      break

  avg_train_loss = total_loss / len(train_loader)

  # Validation
  model.eval()
  val_accuracy = 0
  val_steps = 0

  with torch.no_grad():
    for batch in val_loader:
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
      f"Epoch {epoch + 1}/{num_epochs}, Avg. Training Loss: {avg_train_loss}, Validation Accuracy: {avg_val_accuracy}"
  )

print('Saving trained model')
torch.save(model.state_dict(), './model.pth')