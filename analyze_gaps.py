"""
Multi-Model Analysis Script for NER Models.
Generates confusion matrix grid and confidence histogram comparisons.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.dataset import KeywordDataset
from src.config import ID2LABEL, LABEL2ID, MAX_LEN, get_device


# =============================================================================
# Model Loading (reused from evaluate.py)
# =============================================================================
def load_local_model(model_path: str):
    """
    Load model and tokenizer from a local directory.
    Works around HuggingFace Hub validation issues on Windows.
    """
    model_dir = Path(model_path)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    model_type = config.get("model_type", "distilbert")
    
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
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    
    return tokenizer, model


def load_test_data(path: str) -> list:
    """Load test data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Prediction Functions
# =============================================================================
def get_predictions_and_confidences(model, dataloader, device):
    """
    Run predictions and extract confidence scores.
    
    Returns:
        all_true_ids: List of true label IDs (excluding -100)
        all_pred_ids: List of predicted label IDs
        all_confidences: List of confidence scores for correct predictions
        all_max_probs: List of max probabilities for all predictions
    """
    model.eval()
    all_true_ids = []
    all_pred_ids = []
    all_confidences = []  
    all_max_probs = []    
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            max_probs, predictions = torch.max(probs, dim=-1)
            
            # Process each sample
            for i in range(predictions.shape[0]):
                pred_ids = predictions[i].cpu().numpy()
                true_ids = labels[i].cpu().numpy()
                sample_probs = max_probs[i].cpu().numpy()
                
                for j, (pred_id, true_id, prob) in enumerate(zip(pred_ids, true_ids, sample_probs)):
                    if true_id != -100:  # Ignore special tokens
                        all_true_ids.append(true_id)
                        all_pred_ids.append(pred_id)
                        all_max_probs.append(prob)
                        
                        # Only track confidence for correct predictions
                        if pred_id == true_id:
                            all_confidences.append(prob)
    
    return all_true_ids, all_pred_ids, all_confidences, all_max_probs


# =============================================================================
# Visualization Functions
# =============================================================================
def plot_confusion_matrix_grid(model_results: dict, output_path: str):
    """
    Create a grid of confusion matrices for all models.
    
    Args:
        model_results: Dict mapping model_name -> (true_ids, pred_ids)
        output_path: Path to save the figure.
    """
    n_models = len(model_results)
    labels = list(ID2LABEL.values())  # ["O", "B-KEY", "I-KEY"]
    
    # Set up the figure
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    # Find global max for consistent color scale
    all_cms_normalized = []
    for model_name, (true_ids, pred_ids) in model_results.items():
        cm = confusion_matrix(true_ids, pred_ids, labels=list(ID2LABEL.keys()))
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        all_cms_normalized.append((model_name, cm_normalized, cm))
    
    # Plot each confusion matrix
    for idx, (model_name, cm_normalized, cm_raw) in enumerate(all_cms_normalized):
        ax = axes[idx]
        
        # Create heatmap with normalized values, but show raw counts in annotations
        sns.heatmap(
            cm_normalized,
            annot=cm_raw,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            vmin=0,
            vmax=1,
            cbar=idx == n_models - 1,  # Only show colorbar on last subplot
            square=True
        )
        
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True' if idx == 0 else '', fontsize=10)
        
        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Overall title
    fig.suptitle('Confusion Matrix Comparison (Normalized Colors, Raw Counts)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 Confusion matrix grid saved to: {output_path}")
    plt.show()


def plot_confidence_histogram(model_confidences: dict, output_path: str):
    """
    Create overlapping confidence histograms for all models.
    
    Args:
        model_confidences: Dict mapping model_name -> list of confidence scores
        output_path: Path to save the figure.
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette for models
    colors = sns.color_palette("husl", n_colors=len(model_confidences))
    
    # Plot each model's confidence distribution
    for idx, (model_name, confidences) in enumerate(model_confidences.items()):
        if len(confidences) == 0:
            print(f"⚠️  No correct predictions for {model_name}, skipping histogram")
            continue
            
        sns.kdeplot(
            data=confidences,
            ax=ax,
            label=f'{model_name} (n={len(confidences)}, μ={np.mean(confidences):.3f})',
            color=colors[idx],
            linewidth=2.5,
            fill=True,
            alpha=0.3
        )
    
    # Customize the plot
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score (Probability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Distribution of Correct Predictions by Model', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add vertical reference lines
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Random (0.5)')
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, linewidth=1, label='High Conf (0.9)')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 Confidence histogram saved to: {output_path}")
    plt.show()


# =============================================================================
# Main Analysis Function
# =============================================================================
def analyze_models(model_paths: list, test_data_path: str, output_dir: str):
    """
    Analyze multiple models and generate comparison visualizations.
    """
    device = get_device()
    print(f"Using device: {device}")
    
    # Load test data
    print(f"\n📂 Loading test data from: {test_data_path}")
    test_data = load_test_data(test_data_path)
    print(f"   Test samples: {len(test_data)}")
    
    # Store results for each model
    confusion_results = {}  # model_name -> (true_ids, pred_ids)
    confidence_results = {}  # model_name -> list of confidences
    
    print(f"\n🔍 Analyzing {len(model_paths)} models...\n")
    print("=" * 60)
    
    for model_path in model_paths:
        model_path = os.path.abspath(model_path)
        model_name = os.path.basename(model_path)
        
        print(f"\n📦 Model: {model_name}")
        print(f"   Path: {model_path}")
        
        try:
            # Load model
            tokenizer, model = load_local_model(model_path)
            model.to(device)
            model.eval()
            print(f"   ✅ Model loaded")
            
            # Create dataloader
            dataset = KeywordDataset(
                data=test_data,
                tokenizer=tokenizer,
                label2id=LABEL2ID,
                max_len=MAX_LEN
            )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            # Get predictions and confidences
            true_ids, pred_ids, confidences, _ = get_predictions_and_confidences(
                model, dataloader, device
            )
            
            # Store results
            confusion_results[model_name] = (true_ids, pred_ids)
            confidence_results[model_name] = confidences
            
            # Print summary
            accuracy = sum(t == p for t, p in zip(true_ids, pred_ids)) / len(true_ids)
            avg_conf = np.mean(confidences) if confidences else 0
            print(f"   📊 Token Accuracy: {accuracy:.4f}")
            print(f"   📊 Avg Confidence (correct): {avg_conf:.4f}")
            print(f"   📊 Correct predictions: {len(confidences)}/{len(true_ids)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    print("\n" + "=" * 60)
    
    if not confusion_results:
        print("❌ No models were successfully analyzed.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    
    # 1. Confusion Matrix Grid
    confusion_output = os.path.join(output_dir, "confusion_grid.png")
    plot_confusion_matrix_grid(confusion_results, confusion_output)
    
    # 2. Confidence Histogram
    confidence_output = os.path.join(output_dir, "confidence_comparison.png")
    plot_confidence_histogram(confidence_results, confidence_output)
    
    print("\n✅ Analysis complete!")


# =============================================================================
# Argument Parser
# =============================================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-model analysis for NER: Confusion matrices and confidence histograms"
    )
    
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs='+',
        default=[
            "./output/my_keyword_model",
            "./output/model_llrd",
            "./output/model_adv",
            "./output/model_combo"
        ],
        help="List of model paths to compare"
    )
    
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test_data.json",
        help="Path to test data JSON file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output visualizations"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("       MULTI-MODEL NER ANALYSIS")
    print("=" * 60)
    print(f"📋 Models to analyze: {len(args.model_paths)}")
    for path in args.model_paths:
        print(f"   - {path}")
    print("=" * 60)
    
    analyze_models(
        model_paths=args.model_paths,
        test_data_path=args.test_data,
        output_dir=args.output_dir
    )
