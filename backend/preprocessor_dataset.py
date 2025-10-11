# preprocessor_dataset.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# File paths (backend folder me CSV files ke liye)
FAKE_CSV = "Fake.csv"
TRUE_CSV = "True.csv"
TRAIN_CSV = "train_preprocessed.csv"
TEST_CSV = "test_preprocessed.csv"

class NewsDataset(Dataset):  # <-- note: NewsDataset, NOT News_Dataset
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_dataset(fake_csv=FAKE_CSV, true_csv=TRUE_CSV):
    """Load raw CSV files, add labels, combine, shuffle, split and save preprocessed CSVs."""
    print(f"[INFO] Current working directory: {os.getcwd()}")
    if not os.path.exists(fake_csv) or not os.path.exists(true_csv):
        raise FileNotFoundError(f"Missing CSV files: {fake_csv}, {true_csv}")

    print(f"[INFO] Found CSV file: {fake_csv}")
    print(f"[INFO] Found CSV file: {true_csv}")

    fake_df = pd.read_csv(fake_csv)
    true_df = pd.read_csv(true_csv)

    # Ensure label column exists
    if "label" not in fake_df.columns:
        fake_df["label"] = "FAKE"
    if "label" not in true_df.columns:
        true_df["label"] = "REAL"

    combined_df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42)
    print(f"[INFO] Combined dataset shape: {combined_df.shape}")
    print(combined_df.head())

    # Encode labels
    label_mapping = {"FAKE": np.int64(0), "REAL": np.int64(1)}
    combined_df["label"] = combined_df["label"].map(label_mapping)
    print(f"[INFO] Label encoding mapping: {label_mapping}")

    # Split train/test
    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df["label"])
    print(f"[INFO] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Save preprocessed CSVs
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"[INFO] Preprocessed train/test CSV files saved successfully!")

    return train_df, test_df