"""
Evaluation script for the NER Keyword model.
Computes precision, recall, F1-score using seqeval.
"""

import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DistilBertTokenizerFast, DistilBertForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from src.dataset import KeywordDataset
from src.config import ID2LABEL, LABEL2ID, MAX_LEN, get_device


def load_test_data(path: str) -> list:
    """Load test data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_model(model_path: str):
    """
    Load model and tokenizer from a local directory.
    Works around HuggingFace Hub validation issues on Windows.
    
    Args:
        model_path: Absolute path to the model directory.
    
    Returns:
        Tuple of (tokenizer, model)
    """
    model_dir = Path(model_path)
    
    # Check if directory exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Check for required files
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    
    # Load config to determine model type
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    model_type = config.get("model_type", "distilbert")
    
    # Load tokenizer and model based on type
    if model_type == "distilbert":
        tokenizer = DistilBertTokenizerFast(
            vocab_file=str(model_dir / "vocab.txt"),
            tokenizer_file=str(model_dir / "tokenizer.json"),
        )
        model = DistilBertForTokenClassification.from_pretrained(
            str(model_dir),
            local_files_only=True
        )
    else:
        # Fallback for other model types
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    
    return tokenizer, model


def evaluate(model_path: str, test_data_path: str):
    """
    Evaluate the NER model on the test set.
    
    Args:
        model_path: Path to the trained model directory.
        test_data_path: Path to the test data JSON file.
    """
    # Convert to absolute path
    model_path = os.path.abspath(model_path)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from: {model_path}")
    tokenizer, model = load_local_model(model_path)
    model.to(device)
    model.eval()
    
    # Load test data
    print(f"Loading test data from: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"Test samples: {len(test_data)}")
    
    # Create dataset and dataloader (batch_size=1 for simplicity in alignment)
    test_dataset = KeywordDataset(
        data=test_data,
        tokenizer=tokenizer,
        label2id=LABEL2ID,
        max_len=MAX_LEN
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Accumulate true and predicted labels
    all_true_labels = []
    all_pred_labels = []
    
    print("\nRunning predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get model predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Convert logits to predicted label IDs (argmax)
            predictions = torch.argmax(logits, dim=-1)
            
            # Process each sample in the batch (batch_size=1)
            for i in range(predictions.shape[0]):
                pred_ids = predictions[i].cpu().numpy()
                true_ids = labels[i].cpu().numpy()
                
                # Align predictions with true labels, ignoring -100
                sample_true = []
                sample_pred = []
                
                for pred_id, true_id in zip(pred_ids, true_ids):
                    if true_id != -100:  # Ignore special tokens
                        sample_true.append(ID2LABEL[true_id])
                        sample_pred.append(ID2LABEL[pred_id])
                
                all_true_labels.append(sample_true)
                all_pred_labels.append(sample_pred)
    
    # Calculate metrics
    precision = precision_score(all_true_labels, all_pred_labels)
    recall = recall_score(all_true_labels, all_pred_labels)
    f1 = f1_score(all_true_labels, all_pred_labels)
    
    # Print evaluation results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    # Detailed classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(all_true_labels, all_pred_labels))
    
    # Overall metrics
    print("-" * 60)
    print("Overall Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print("=" * 60)
    
    # Save metrics to JSON file
    model_name = os.path.basename(os.path.normpath(model_path))
    metrics = {
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "model_name": model_name
    }
    
    metrics_path = os.path.join(model_path, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n📄 Metrics saved to: {metrics_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NER model on test data"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/my_keyword_model",
        help="Path to the trained model directory (default: ./output/my_keyword_model)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    MODEL_PATH = args.model_path
    TEST_DATA_PATH = "./data/test_data.json"
    
    print(f"🔍 Evaluating model at: {MODEL_PATH}...")
    evaluate(MODEL_PATH, TEST_DATA_PATH)
