# bert_trainer.py
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_scheduler  # AdamW included internally in optimizer
from preprocessor_dataset import NewsDataset  # corrected class
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Paths to preprocessed CSVs
TRAIN_CSV = "train_preprocessed.csv"
TEST_CSV = "test_preprocessed.csv"

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

# Load preprocessed datasets
logger.info("Loading preprocessed datasets...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# Load tokenizer
logger.info("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare datasets
train_dataset = NewsDataset(train_df['text'], train_df['label'], tokenizer, MAX_LEN)
test_dataset = NewsDataset(test_df['text'], test_df['label'], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Load model
logger.info("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model = model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training loop
logger.info("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

# Save trained model
MODEL_PATH = "bert_fake_news_model"
logger.info(f"Saving trained model to {MODEL_PATH}...")
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
logger.info("Training complete!")