import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths relative to project root (~/projects/AGERE/)
PARSED_DIR = "results/training/tb_parsed"
OUT_DIR = "results/training/aggregated"
PLOT_DIR = "inspector/plots/training_curves"

def main():
    # Ensure output directories exist
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 1. Load all parsed CSVs
    csv_files = glob.glob(os.path.join(PARSED_DIR, "*_scalars.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {PARSED_DIR}")
        return
    
    print(f"Loading {len(csv_files)} seed files...")
    dfs = []
    for f in csv_files:
        seed_name = os.path.basename(f).replace("_scalars.csv", "")
        df = pd.read_csv(f)
        df['seed'] = seed_name
        dfs.append(df)
        
    # Combine into one dataframe
    df_all = pd.concat(dfs, ignore_index=True)
    
    # FIX: The CSVs are in wide format (tags are columns). We need to melt them 
    # back into long format (step, seed, tag, value) for seaborn plotting.
    id_vars = ['step', 'seed']
    value_vars = [col for col in df_all.columns if col not in id_vars]
    df_long = df_all.melt(id_vars=id_vars, value_vars=value_vars, var_name='tag', value_name='value')
    
    # Save aggregated raw data for reproducibility (reviewers love this)
    agg_path = os.path.join(OUT_DIR, "all_seeds_training.csv")
    df_long.to_csv(agg_path, index=False)
    print(f"Saved aggregated data to: {agg_path}")
    
    # 2. Define the key metrics to plot for the paper
    metrics_to_plot = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean',
        'train/value_loss',
        'train/entropy_loss'
    ]
    
    # Filter dataframe to only these metrics
    df_plot = df_long[df_long['tag'].isin(metrics_to_plot)].copy()
    
    # 3. Configure plotting style for academic papers (clean, readable)
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
            color="#1f77b4" # Professional blue
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

