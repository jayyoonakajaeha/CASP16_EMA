import pandas as pd
import glob
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge individual target prediction CSVs into a single file for evaluation.")
    parser.add_argument("--pred_dir", required=True, help="Directory containing prediction CSV files (e.g. T1104.csv)")
    parser.add_argument("--score_dir", help="Directory containing Ground Truth Score CSVs (Optional). If provided, merges scores.", default=None)
    parser.add_argument("--output", required=True, help="Output merged CSV filename")
    
    args = parser.parse_args()
    
    # Pattern: pred_dir/*.csv
    pred_files = glob.glob(os.path.join(args.pred_dir, "*.csv"))
    if not pred_files:
        print(f"No CSV files found in {args.pred_dir}")
        return

    merged_data = []
    
    print(f"Found {len(pred_files)} prediction files.")
    
    for f in pred_files:
        target = os.path.basename(f).replace('.csv', '')
        # Special handling for target name if needed?
        
        try:
            df = pd.read_csv(f)
            # Expect 'model' column or similar
            # Standardize Model column
            if 'model' in df.columns:
                df.rename(columns={'model': 'Model'}, inplace=True)
            elif 'Model' not in df.columns:
                print(f"Warning: {f} has no 'model' or 'Model' column. Skipping.")
                continue
                
            df['Target'] = target
            
            # If ground truth logic is needed:
            if args.score_dir:
                score_file = os.path.join(args.score_dir, f"{target}_quality_scores.csv")
                if os.path.exists(score_file):
                    score_df = pd.read_csv(score_file)
                    # Merge logic (assuming standard format matching load_data)
                    # Ensure merge keys
                    if 'model_name' in score_df.columns:
                        score_df['merge_key'] = score_df['model_name'].astype(str).str.replace('.pdb', '').str.strip()
                        df['merge_key'] = df['Model'].astype(str).str.strip()
                        
                        # Merge only score metric columns + merge_key
                        score_cols = [c for c in score_df.columns if c not in ['model_name', 'merge_key']]
                        score_df_subset = score_df[['merge_key'] + score_cols]
                        
                        df = pd.merge(df, score_df_subset, on='merge_key', how='inner')
                        df.drop(columns=['merge_key'], inplace=True)
            
            merged_data.append(df)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if merged_data:
        full_df = pd.concat(merged_data, ignore_index=True)
        full_df.to_csv(args.output, index=False)
        print(f"Successfully merged {len(full_df)} rows. Saved to {args.output}")
        print("Columns:", list(full_df.columns))
    else:
        print("No data merged.")

if __name__ == "__main__":
    main()
