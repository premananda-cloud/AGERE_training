import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths (adjust if your CSV is saved elsewhere)
INPUT_CSV = "results/training/aggregated/all_seeds_training.csv"
PLOT_DIR = "inspector/plots/training_curves"

def main():
    # Ensure output directory exists
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Load the aggregated long-format CSV
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Could not find {INPUT_CSV}")
        print("Please ensure the parsing script was run and the file is in the correct location.")
        return
        
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 2. Define the key metrics to plot for the paper
    # These directly address the reviewer's request for training stability and variability
    metrics_to_plot = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean',
        'train/value_loss',
        'train/entropy_loss'
    ]
    
    # Filter dataframe to only these metrics
    df_plot = df[df['tag'].isin(metrics_to_plot)].copy()
    
    # 3. Configure plotting style for academic papers (clean, readable, no clutter)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    print("\nGenerating publication-ready plots...")
    for metric in metrics_to_plot:
        df_metric = df_plot[df_plot['tag'] == metric]
        
        if df_metric.empty:
            print(f"  WARNING: No data found for {metric}")
            continue
            
        plt.figure(figsize=(8, 5))
        
        # seaborn lineplot automatically calculates the mean and 95% Confidence Interval
        # The shaded region is the visual proof of low variance across your 10 seeds
        sns.lineplot(
            data=df_metric,
            x='step',
            y='value',
            errorbar=('ci', 95),  # 95% confidence interval
            linewidth=2.5,
            color="#1f77b4" # Professional academic blue
        )
        
        # Clean up the title for readability
        clean_name = metric.replace('rollout/', '').replace('train/', '').replace('_', ' ').title()
        plt.title(f"Training Stability: {clean_name} (10 Seeds)", fontsize=14)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        
        # Save as high-res PDF (best for LaTeX) and PNG
        base_name = metric.replace('/', '_')
        pdf_path = os.path.join(PLOT_DIR, f"{base_name}_curve.pdf")
        png_path = os.path.join(PLOT_DIR, f"{base_name}_curve.png")
        
        plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
        plt.savefig(png_path, bbox_inches='tight', dpi=300)
        print(f"  Saved: {pdf_path}")
        
        plt.close()

    print("\n✅ Phase 1 Plotting Complete! Check the inspector/plots/training_curves/ directory.")

if __name__ == "__main__":
    main()
