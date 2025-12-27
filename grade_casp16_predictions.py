import pandas as pd
import numpy as np
import os
import glob
import argparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

def calculate_metrics(df, pred_col, truth_col='tmscore_mmalign'):
    # Prepare data
    # Drop rows where either Prediction or Truth is missing
    clean_df = df.dropna(subset=[pred_col, truth_col])
    
    if len(clean_df) < 5: # Too few samples
        return {'Pearson': np.nan, 'Spearman': np.nan, 'Loss': np.nan, 'AUROC': np.nan}
    
    y_pred = clean_df[pred_col]
    y_true = clean_df[truth_col]
    
    # 80% Rule Check
    # Total models for target = len(df)
    # Submitted models = len(clean_df)
    # If submitted / total < 0.8, penalty?
    # User said: "80% rule" in context of previous task.
    # CASP rule: If coverage < X%, Z-score = 0 (or -2).
    # Here we just output NaNs if insufficient coverage, and let Z-score script handle it (treating NaNs as mean or 0).
    # Actually, previous instruction said: "If < 80%, treat metrics as missing (score 0)".
    # So returning NaNs is appropriate.
    
    coverage = len(clean_df) / len(df)
    if coverage < 0.8:
        # print(f"  coverage {coverage:.2f} < 0.8")
        return {'Pearson': np.nan, 'Spearman': np.nan, 'Loss': np.nan, 'AUROC': np.nan}

    # Target Quality Filter (CASP16 Rule)
    # If the best model in the pool has quality < 0.6, exclude this target from evaluation.
    max_true_val = y_true.max()
    if max_true_val < 0.6:
        # print(f"  Target max quality {max_true_val:.3f} < 0.6. Skipping.")
        return {'Pearson': np.nan, 'Spearman': np.nan, 'Loss': np.nan, 'AUROC': np.nan}

    # Pearson
    p_r, _ = pearsonr(y_pred, y_true)
    
    # Spearman
    s_r, _ = spearmanr(y_pred, y_true)
    
    # Loss
    # Identify top predicted model(s)
    max_pred_val = y_pred.max()
    top_models = clean_df[clean_df[pred_col] == max_pred_val]
    # Corresponding true scores
    top_true_scores = top_models[truth_col]
    
    avg_top_true = top_true_scores.mean()
    loss = max_true_val - avg_top_true
    
    # AUROC
    # CASP16 Rule: "Good" models are top 25% (75th percentile) of the pool.
    # Relative thresholding.
    threshold = np.percentile(y_true, 75)
    
    # If standard deviation is very small, or many ties, threshold might capture all or none?
    # np.percentile handles it.
    # y_class = 1 if score >= threshold
    y_class = (y_true >= threshold).astype(int)
    
    # Check if we have both classes
    if len(np.unique(y_class)) < 2:
        auroc = np.nan
    else:
        try:
            auroc = roc_auc_score(y_class, y_pred)
        except:
            auroc = np.nan
            
    return {'Pearson': p_r, 'Spearman': s_r, 'Loss': loss, 'AUROC': auroc}

def main():
    parser = argparse.ArgumentParser(description="Grade Community EMA Predictions")
    parser.add_argument("--pred_dir", required=True, help="Directory with Prediction CSVs")
    parser.add_argument("--score_dir", required=True, help="Directory with Quality Score CSVs")
    parser.add_argument("--output", required=True, help="Output graded CSV")
    parser.add_argument("--truth_metric", default="tmscore_mmalign", help="Name of the ground truth metric column (default: tmscore_mmalign)")
    parser.add_argument("--user_csv", help="Optional: Path to your own model's predictions (CSV with Target, Model, Score)")
    parser.add_argument("--user_name", default="MyModel", help="Name of your model (for the leaderboard)")
    
    args = parser.parse_args()
    
    pred_files = glob.glob(os.path.join(args.pred_dir, "*.csv"))
    all_results = []
    
    print(f"Found {len(pred_files)} prediction files.")
    print(f"Using ground truth metric: {args.truth_metric}")
    
    # Load User CSV if provided
    user_df = None
    if args.user_csv:
        if os.path.exists(args.user_csv):
            print(f"Loading user predictions from {args.user_csv}...")
            try:
                user_df = pd.read_csv(args.user_csv)
                # Check columns
                required_cols = ['Target', 'Model', 'Score']
                # If they use slightly different, maybe warn?
                # User instruction: Make sure CSV has Target, Model, Score columns.
                if not all(col in user_df.columns for col in required_cols):
                     print(f"Error: User CSV must contain columns: {required_cols}")
                     return
            except Exception as e:
                print(f"Error reading user CSV: {e}")
                return
        else:
            print(f"Error: User CSV {args.user_csv} not found.")
            return

    for pf in pred_files:
        target = os.path.basename(pf).replace('.csv', '')
        # Special handling for versions (v1o, v2o)?
        # Quality scores have similar names.
        
        qf = os.path.join(args.score_dir, f"{target}_quality_scores.csv")
        if not os.path.exists(qf):
            print(f"Skipping {target}: Quality score file not found.")
            continue
            
        try:
            # Read Data
            df_pred = pd.read_csv(pf)
            df_true = pd.read_csv(qf)
            
            # Check if truth metric exists
            if args.truth_metric not in df_true.columns:
                print(f"Skipping {target}: Metric '{args.truth_metric}' not found in quality scores.")
                continue

            # Merge
            df_true['join_id'] = df_true['model_name'].str.replace('.pdb', '', regex=False)
            df_pred['join_id'] = df_pred['model'] # Assuming 'model' col exists
            
            # --- Inject User Model if provided ---
            if args.user_csv and os.path.exists(args.user_csv):
                try:
                    # User CSV format assumed: Target, Decoy, Score (or Model, Score if standard format)
                    # Ideally: Target_ID (e.g. H1202), Model_ID (e.g. H1202TS014_1), Score
                    # Let's support a simple format: "Target,Model,Score"
                    
                    # Optimization: Load user_df once outside loop? 
                    # For simplicity and memory safety with huge files, we load once globally. 
                    # But here we are inside loop. Let's do lazy load or assume global load.
                    # Implementing global load outside loop is better.
                    pass 
                except Exception as e:
                    print(f"Warning: Failed to process user CSV for {target}: {e}")

            # Merge with user data if available in pre-loaded df
            if user_df is not None:
                # Filter for this target
                # User CSV might have 'Target' column. 
                # Model IDs in user CSV must match df_pred['join_id']
                
                # Check if 'Target' column exists or if we rely on Model ID matching
                # Safer to filter by Target if available to avoid ID collisions?
                # Let's assume Model_ID is unique enough or user provides relevant rows.
                
                user_subset = user_df[user_df['Model'].isin(df_pred['join_id'])].copy()
                
                if not user_subset.empty:
                    # Rename score column to args.user_name
                    user_subset = user_subset[['Model', 'Score']].rename(columns={'Score': args.user_name, 'Model': 'join_id'})
                    # Merge to df_pred
                    df_pred = pd.merge(df_pred, user_subset, on='join_id', how='left')
                    # Fill NaNs? If user didn't predict some decoys, they stay NaN -> Coverage check will handle it.
            
            merged = pd.merge(df_pred, df_true[['join_id', args.truth_metric]], on='join_id', how='inner')
            
            if len(merged) == 0:
                print(f"Warning: {target} merge resulted in 0 rows. Check IDs.")
                continue
                
            # Iterate Predictors
            # Predictors are all columns in df_pred except 'model' and 'join_id'
            predictors = [c for c in df_pred.columns if c not in ['model', 'join_id']]
            
            for pred in predictors:
                metrics = calculate_metrics(merged, pred, args.truth_metric)
                # metrics dict: Pearson, Spearman, Loss, AUROC
                
                res = {'Target': target, 'Model': pred}
                res.update(metrics)
                all_results.append(res)
                
        except Exception as e:
            print(f"Error processing {target}: {e}")
            
    # Save
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv(args.output, index=False)
        print(f"Saved graded metrics to {args.output}")
        # Print summary
        print("Models Graded:", final_df['Model'].nunique())
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
