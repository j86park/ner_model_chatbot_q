"""
Compare multiple NER experiments by visualizing their metrics.
Creates a grouped bar chart comparing F1, Precision, and Recall across models.
"""

import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# =============================================================================
# CONFIGURATION: List of experiment folder names to compare
# =============================================================================
EXPERIMENTS = [
    "my_keyword_model",
    "model_llrd",
    "model_adv",
    "model_combo"
]

# Set to True to re-run evaluate.py before loading metrics (ensures fresh data)
RUN_EVALUATION = False

# Base output directory
OUTPUT_DIR = "./output"


def run_evaluation(experiment_name: str) -> bool:
    """
    Run evaluate.py for a specific experiment to generate fresh metrics.
    
    Args:
        experiment_name: Name of the experiment folder.
    
    Returns:
        True if evaluation succeeded, False otherwise.
    """
    model_path = os.path.join(OUTPUT_DIR, experiment_name)
    
    if not os.path.exists(model_path):
        print(f"⚠️  Skipping {experiment_name}: folder not found")
        return False
    
    print(f"🔄 Running evaluation for: {experiment_name}")
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--model_path", model_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Evaluation failed for {experiment_name}")
        print(result.stderr)
        return False
    
    return True


def load_metrics(experiment_name: str) -> dict | None:
    """
    Load metrics.json from an experiment folder.
    
    Args:
        experiment_name: Name of the experiment folder.
    
    Returns:
        Dictionary with metrics or None if not found.
    """
    metrics_path = os.path.join(OUTPUT_DIR, experiment_name, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"⚠️  No metrics.json found for: {experiment_name}")
        return None
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_comparison_chart(df: pd.DataFrame, output_path: str):
    """
    Create a grouped bar chart comparing model metrics.
    
    Args:
        df: DataFrame with columns [Model, Metric, Score].
        output_path: Path to save the chart image.
    """
    # Set up the style
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette
    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # Green, Blue, Red
    
    # Create grouped bar chart
    bar_plot = sns.barplot(
        data=df,
        x="Model",
        y="Score",
        hue="Metric",
        palette=colors,
        ax=ax,
        edgecolor="white",
        linewidth=1.5
    )
    
    # Add data labels on bars
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            label_type="edge",
            fontsize=10,
            fontweight="bold",
            padding=3
        )
    
    # Customize the chart
    ax.set_ylim(0, 1.15)  # Extra space for labels
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title(
        "NER Model Performance Comparison",
        fontsize=16,
        fontweight="bold",
        pad=20
    )
    
    # Customize legend
    ax.legend(
        title="Metric",
        title_fontsize=11,
        fontsize=10,
        loc="upper right",
        framealpha=0.9
    )
    
    # Add horizontal reference lines
    ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.3, linewidth=1)
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=15, ha="right")
    
    # Tight layout
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"📊 Chart saved to: {output_path}")
    
    # Show the chart
    plt.show()


def main():
    print("\n" + "=" * 60)
    print("       NER EXPERIMENT COMPARISON")
    print("=" * 60)
    print(f"📋 Experiments to compare: {EXPERIMENTS}")
    print("=" * 60 + "\n")
    
    # Optionally run evaluations first
    if RUN_EVALUATION:
        print("🔄 Running evaluations to ensure fresh metrics...\n")
        for exp in EXPERIMENTS:
            run_evaluation(exp)
        print()
    
    # Collect metrics from all experiments
    all_metrics = []
    
    for exp_name in EXPERIMENTS:
        metrics = load_metrics(exp_name)
        if metrics:
            all_metrics.append({
                "model_name": metrics.get("model_name", exp_name),
                "f1": metrics.get("f1", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
            })
            print(f"✅ Loaded metrics for: {exp_name}")
    
    if not all_metrics:
        print("\n❌ No metrics found. Please run evaluate.py first.")
        return
    
    # Create DataFrame in long format for seaborn
    records = []
    for m in all_metrics:
        records.append({"Model": m["model_name"], "Metric": "F1", "Score": m["f1"]})
        records.append({"Model": m["model_name"], "Metric": "Precision", "Score": m["precision"]})
        records.append({"Model": m["model_name"], "Metric": "Recall", "Score": m["recall"]})
    
    df = pd.DataFrame(records)
    
    # Print summary table
    print("\n" + "-" * 60)
    print("METRICS SUMMARY")
    print("-" * 60)
    summary_df = pd.DataFrame(all_metrics)
    summary_df.columns = ["Model", "F1", "Precision", "Recall"]
    print(summary_df.to_string(index=False))
    print("-" * 60 + "\n")
    
    # Create and save the comparison chart
    output_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    create_comparison_chart(df, output_path)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()
