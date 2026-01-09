import json
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src import KeywordDataset
from src import config


def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def train_epoch(model, dataloader, optimizer, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def main():
    # Get device
    device = config.get_device()
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    data = load_data("data/raw_data.json")
    print(f"Loaded {len(data)} samples")

    # Split data: 80% train, 20% validation
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Load tokenizer
    print(f"Loading tokenizer: {config.MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)

    # Create datasets
    train_dataset = KeywordDataset(
        data=train_data,
        tokenizer=tokenizer,
        label2id=config.LABEL2ID,
        max_len=config.MAX_LEN,
    )
    val_dataset = KeywordDataset(
        data=val_data,
        tokenizer=tokenizer,
        label2id=config.LABEL2ID,
        max_len=config.MAX_LEN,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    # Load model with label configuration
    print(f"Loading model: {config.MODEL_CHECKPOINT}")
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=len(config.LABEL2ID),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Training loop
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print("-" * 50)

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")

        # Training phase
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}")

        # Validation phase
        val_loss = validate_epoch(model, val_loader, device)
        print(f"  Validation Loss: {val_loss:.4f}")

    print("-" * 50)
    print("Training complete!")

    # Save model, tokenizer, and config
    output_dir = "./output/my_keyword_model"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model and tokenizer saved successfully!")


if __name__ == "__main__":
    main()

