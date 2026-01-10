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


def checkpoint(step, message):
    """Print a formatted checkpoint message."""
    print(f"\n{'='*60}")
    print(f"[CHECKPOINT {step}] {message}")
    print(f"{'='*60}")


def main():
    print("\n" + "=" * 60)
    print("       NER MODEL TRAINING PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # CHECKPOINT 1: Device Configuration
    # -------------------------------------------------------------------------
    checkpoint(1, "DEVICE CONFIGURATION")
    device = config.get_device()
    print(f"  -> Device selected: {device}")
    print(f"  -> CUDA available: {torch.cuda.is_available()}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 2: Data Loading
    # -------------------------------------------------------------------------
    checkpoint(2, "LOADING RAW DATA")
    data = load_data("data/raw_data.json")
    print(f"  -> Successfully loaded {len(data)} samples from data/raw_data.json")

    # -------------------------------------------------------------------------
    # CHECKPOINT 3: Train/Validation Split
    # -------------------------------------------------------------------------
    checkpoint(3, "SPLITTING DATA (80/20)")
    train_data, val_data = train_test_split(
        data, test_size=0.2, random_state=42
    )
    print(f"  -> Training samples:   {len(train_data)}")
    print(f"  -> Validation samples: {len(val_data)}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 4: Tokenizer Initialization
    # -------------------------------------------------------------------------
    checkpoint(4, "LOADING TOKENIZER")
    print(f"  -> Model checkpoint: {config.MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    print(f"  -> Tokenizer loaded successfully")
    print(f"  -> Vocab size: {tokenizer.vocab_size}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 5: Dataset Creation
    # -------------------------------------------------------------------------
    checkpoint(5, "CREATING PYTORCH DATASETS")
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
    print(f"  -> Train dataset: {len(train_dataset)} samples")
    print(f"  -> Val dataset:   {len(val_dataset)} samples")
    print(f"  -> Max sequence length: {config.MAX_LEN}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 6: DataLoader Creation
    # -------------------------------------------------------------------------
    checkpoint(6, "CREATING DATALOADERS")
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    print(f"  -> Batch size: {config.BATCH_SIZE}")
    print(f"  -> Train batches per epoch: {len(train_loader)}")
    print(f"  -> Val batches per epoch:   {len(val_loader)}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 7: Model Initialization
    # -------------------------------------------------------------------------
    checkpoint(7, "LOADING PRE-TRAINED MODEL")
    print(f"  -> Loading: {config.MODEL_CHECKPOINT}")
    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=len(config.LABEL2ID),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    model.to(device)
    print(f"  -> Model loaded and moved to {device}")
    print(f"  -> Number of labels: {len(config.LABEL2ID)}")
    print(f"  -> Labels: {list(config.LABEL2ID.keys())}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 8: Optimizer Setup
    # -------------------------------------------------------------------------
    checkpoint(8, "CONFIGURING OPTIMIZER")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    print(f"  -> Optimizer: AdamW")
    print(f"  -> Learning rate: {config.LEARNING_RATE}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 9: Training Loop
    # -------------------------------------------------------------------------
    checkpoint(9, "STARTING TRAINING LOOP")
    print(f"  -> Total epochs: {config.EPOCHS}")
    print("-" * 60)

    for epoch in range(config.EPOCHS):
        print(f"\n>>> EPOCH {epoch + 1}/{config.EPOCHS}")

        # Training phase
        print("  [Training phase...]")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"  -> Train Loss: {train_loss:.4f}")

        # Validation phase
        print("  [Validation phase...]")
        val_loss = validate_epoch(model, val_loader, device)
        print(f"  -> Validation Loss: {val_loss:.4f}")

        print(f"  [Epoch {epoch + 1} complete]")

    # -------------------------------------------------------------------------
    # CHECKPOINT 10: Save Model
    # -------------------------------------------------------------------------
    checkpoint(10, "SAVING MODEL & TOKENIZER")
    output_dir = "./output/my_keyword_model"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  -> Output directory: {output_dir}")

    model.save_pretrained(output_dir)
    print(f"  -> Model saved")

    tokenizer.save_pretrained(output_dir)
    print(f"  -> Tokenizer saved")

    # -------------------------------------------------------------------------
    # FINAL: Pipeline Complete
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("       TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"  Model saved to: {output_dir}")
    print(f"  Ready for inference!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

