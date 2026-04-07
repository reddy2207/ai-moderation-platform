import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch

# ----------------------------
# LOAD DATA
# ----------------------------

df = pd.read_csv("../data/final_dataset.csv")

# 🔥 FIX: clean bad rows
df = df.dropna(subset=["clean_text"])
df["clean_text"] = df["clean_text"].astype(str)

# 🔥 LIMIT DATA SIZE (FAST TRAINING)
df = df.sample(n=20000, random_state=42)

print("Dataset loaded and optimized ✅")

# ----------------------------
# ENCODE LABELS
# ----------------------------

label_map = {
    "safe": 0,
    "toxic": 1,
    "abuse": 2,
    "hate": 3,
    "other": 4
}

df["label"] = df["label"].map(label_map)

# ----------------------------
# TRAIN-TEST SPLIT
# ----------------------------

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_text"], df["label"], test_size=0.1
)

# ----------------------------
# TOKENIZATION
# ----------------------------

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# ----------------------------
# DATASET CLASS
# ----------------------------

class ModerationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ModerationDataset(train_encodings, train_labels)
val_dataset = ModerationDataset(val_encodings, val_labels)

# ----------------------------
# LOAD MODEL
# ----------------------------

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=5
)

# 🔥 USE APPLE GPU (MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# ----------------------------
# TRAINING SETTINGS (OPTIMIZED)
# ----------------------------

training_args = TrainingArguments(
    output_dir="../models",
    learning_rate=2e-5,
    per_device_train_batch_size=8,   # reduced load
    per_device_eval_batch_size=8,
    num_train_epochs=1,              # faster training
    weight_decay=0.01,
    logging_dir="../logs",
    eval_strategy="epoch",
    save_strategy="epoch"
)

# ----------------------------
# TRAINER
# ----------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ----------------------------
# TRAIN MODEL
# ----------------------------

print("Training started 🚀")
trainer.train()

# ----------------------------
# SAVE MODEL
# ----------------------------

model.save_pretrained("../models/moderation_model")
tokenizer.save_pretrained("../models/moderation_model")

print("Model saved ✅")