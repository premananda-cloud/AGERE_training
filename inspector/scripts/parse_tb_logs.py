import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Paths relative to the project root (~/projects/AGERE/)
TB_BASE_DIR = "tb_logs/hover_logs"
OUT_DIR = "results/training/tb_parsed"

def load_tb_data(log_dir):
    """Loads TensorBoard event data from a directory."""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    return ea

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Find all PPO directories and filter for our 10 seed study (PPO_10 to PPO_19)
    all_dirs = sorted(glob.glob(os.path.join(TB_BASE_DIR, "PPO_*")))
    seed_dirs = [d for d in all_dirs if any(f"PPO_{i}" in d for i in range(10, 20))]
    
    if not seed_dirs:
        print("ERROR: No seed directories found between PPO_10 and PPO_19.")
        return

    all_tags = set()
    seed_data = {}
    
    print("="*50)
    print("DISCOVERING TENSORBOARD SCALARS")
    print("="*50)
    
    # 1. Discover what metrics exist
    for s_dir in seed_dirs:
        seed_name = os.path.basename(s_dir)
        ea = load_tb_data(s_dir)
        tags = ea.Tags().get('scalars', [])
        all_tags.update(tags)
        seed_data[seed_name] = ea
        print(f"  {seed_name}: {len(tags)} scalars found.")
        
    print("\n--- ALL AVAILABLE SCALAR TAGS ---")
    for tag in sorted(list(all_tags)):
        print(f"  - {tag}")
    print("---------------------------------\n")
    
    # 2. Extract data into clean CSVs
    print("="*50)
    print("EXTRACTING DATA TO CSV")
    print("="*50)
    
    for seed_name, ea in seed_data.items():
        tags = ea.Tags().get('scalars', [])
        data = []
        
        for tag in tags:
            events = ea.Scalars(tag)
            for e in events:
                data.append({'step': e.step, 'tag': tag, 'value': e.value})
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Pivot so that 'step' is the index and each 'tag' is a column
            df_pivot = df.pivot_table(index='step', columns='tag', values='value')
            out_path = os.path.join(OUT_DIR, f"{seed_name}_scalars.csv")
            df_pivot.to_csv(out_path)
            print(f"Saved: {out_path} ({len(df_pivot)} steps, {len(df_pivot.columns)} metrics)")
        else:
            print(f"WARNING: No data extracted for {seed_name}")

    print("\nPhase 1 Parsing Complete. Please review the tags printed above.")

if __name__ == "__main__":
    main()
