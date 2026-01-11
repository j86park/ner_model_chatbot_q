"""
Advanced Training Script for NER Model.
Supports Layer-wise Learning Rate Decay (LLRD) and FGM Adversarial Training.
"""

import argparse
import json
import os
import re

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src import KeywordDataset
from src import config


# =============================================================================
# FGM (Fast Gradient Method) Adversarial Training
# =============================================================================
class FGM:
    """
    Fast Gradient Method for adversarial training.
    Adds perturbations to word embeddings to improve model robustness.
    """

    def __init__(self, model, epsilon=1.0, emb_name="word_embeddings"):
        """
        Args:
            model: The model to apply adversarial training to.
            epsilon: Perturbation magnitude.
            emb_name: Name of the embedding layer to perturb.
        """
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """
        Add adversarial perturbation to embeddings.
        Call this after loss.backward().
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                # Backup original embeddings
                self.backup[name] = param.data.clone()
                # Compute perturbation
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # r_adv = epsilon * grad / ||grad||
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self):
        """
        Restore original embeddings after adversarial step.
        Call this after adversarial backward pass.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# LLRD (Layer-wise Learning Rate Decay)
# =============================================================================
def get_optimizer_grouped_parameters(model, base_lr, weight_decay=0.01, layer_decay=0.95):
    """
    Create optimizer parameter groups with layer-wise learning rate decay.
    Lower layers get smaller learning rates (decaying by layer_decay factor).
    
    Args:
        model: The transformer model.
        base_lr: Base learning rate for the top layer.
        weight_decay: Weight decay for regularization.
        layer_decay: Decay factor for each layer (e.g., 0.95).
    
    Returns:
        List of parameter groups for the optimizer.
    """
    # No weight decay for bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    
    # Get the number of layers
    # DistilBERT has 6 layers, BERT-base has 12
    num_layers = 6  # DistilBERT default
    
    # Try to detect actual number of layers
    for name, _ in model.named_parameters():
        match = re.search(r"layer\.(\d+)\.", name)
        if match:
            layer_num = int(match.group(1))
            num_layers = max(num_layers, layer_num + 1)
    
    # Layer groups: embeddings -> transformer layers -> classifier
    layer_scales = {}
    
    # Embeddings get the lowest LR
    layer_scales["embeddings"] = layer_decay ** (num_layers + 1)
    
    # Each transformer layer gets progressively higher LR
    for layer_idx in range(num_layers):
        layer_scales[f"layer.{layer_idx}."] = layer_decay ** (num_layers - layer_idx)
    
    # Classifier head gets the base LR
    layer_scales["classifier"] = 1.0
    layer_scales["pre_classifier"] = 1.0  # DistilBERT has this
    
    optimizer_grouped_parameters = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Determine the layer scale for this parameter
        lr_scale = 1.0
        for key, scale in layer_scales.items():
            if key in name:
                lr_scale = scale
                break
        
        # Determine weight decay
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        
        optimizer_grouped_parameters.append({
            "params": [param],
            "lr": base_lr * lr_scale,
            "weight_decay": wd,
        })
    
    return optimizer_grouped_parameters


# =============================================================================
# Data Loading
# =============================================================================
def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Training Functions
# =============================================================================
def train_epoch(model, dataloader, optimizer, device, fgm=None):
    """
    Run one training epoch.
    
    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: The optimizer.
        device: Device to run on.
        fgm: Optional FGM instance for adversarial training.
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # FGM Adversarial Training
        if fgm is not None:
            # Attack: add perturbation to embeddings
            fgm.attack()
            
            # Adversarial forward pass
            adv_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            adv_loss = adv_outputs.loss
            
            # Adversarial backward pass
            adv_loss.backward()
            
            # Restore original embeddings
            fgm.restore()

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


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced NER Training with LLRD and FGM support"
    )
    
    parser.add_argument(
        "--exp_name",
        type=str,
        default="baseline",
        help="Experiment name. Output will be saved to output/{exp_name}"
    )
    
    parser.add_argument(
        "--use_llrd",
        action="store_true",
        help="Enable Layer-wise Learning Rate Decay"
    )
    
    parser.add_argument(
        "--use_fgm",
        action="store_true",
        help="Enable FGM Adversarial Training"
    )
    
    return parser.parse_args()


# =============================================================================
# Main Training Pipeline
# =============================================================================
def main():
    args = parse_args()
    
    # Build experiment banner
    techniques = []
    if args.use_llrd:
        techniques.append("LLRD")
    if args.use_fgm:
        techniques.append("FGM")
    
    technique_str = " + ".join(techniques) if techniques else "Baseline"
    
    print("\n" + "=" * 60)
    print("       ADVANCED NER MODEL TRAINING PIPELINE")
    print("=" * 60)
    print(f"🚀 Starting Experiment: {args.exp_name}")
    print(f"📋 Active Techniques: {technique_str}")
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
    
    if args.use_llrd:
        print(f"  -> Using Layer-wise Learning Rate Decay (LLRD)")
        print(f"  -> Layer decay factor: 0.95")
        optimizer_params = get_optimizer_grouped_parameters(
            model=model,
            base_lr=config.LEARNING_RATE,
            weight_decay=0.01,
            layer_decay=0.95
        )
        optimizer = torch.optim.AdamW(optimizer_params)
        print(f"  -> Created {len(optimizer_params)} parameter groups")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        print(f"  -> Using standard AdamW optimizer")
    
    print(f"  -> Base learning rate: {config.LEARNING_RATE}")

    # -------------------------------------------------------------------------
    # CHECKPOINT 9: FGM Setup (if enabled)
    # -------------------------------------------------------------------------
    fgm = None
    if args.use_fgm:
        checkpoint(9, "CONFIGURING FGM ADVERSARIAL TRAINING")
        fgm = FGM(model, epsilon=1.0, emb_name="word_embeddings")
        print(f"  -> FGM epsilon: 1.0")
        print(f"  -> Target: word_embeddings layer")
    else:
        checkpoint(9, "SKIPPING FGM (not enabled)")
        print(f"  -> FGM adversarial training: DISABLED")

    # -------------------------------------------------------------------------
    # CHECKPOINT 10: Training Loop
    # -------------------------------------------------------------------------
    checkpoint(10, "STARTING TRAINING LOOP")
    print(f"  -> Total epochs: {config.EPOCHS}")
    print(f"  -> Techniques: {technique_str}")
    print("-" * 60)

    for epoch in range(config.EPOCHS):
        print(f"\n>>> EPOCH {epoch + 1}/{config.EPOCHS}")

        # Training phase
        print("  [Training phase...]")
        train_loss = train_epoch(model, train_loader, optimizer, device, fgm=fgm)
        print(f"  -> Train Loss: {train_loss:.4f}")

        # Validation phase
        print("  [Validation phase...]")
        val_loss = validate_epoch(model, val_loader, device)
        print(f"  -> Validation Loss: {val_loss:.4f}")

        print(f"  [Epoch {epoch + 1} complete]")

    # -------------------------------------------------------------------------
    # CHECKPOINT 11: Save Model
    # -------------------------------------------------------------------------
    checkpoint(11, "SAVING MODEL & TOKENIZER")
    output_dir = f"./output/{args.exp_name}"
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
    print(f"  🎯 Experiment: {args.exp_name}")
    print(f"  📋 Techniques: {technique_str}")
    print(f"  📁 Model saved to: {output_dir}")
    print(f"  ✅ Ready for inference!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
