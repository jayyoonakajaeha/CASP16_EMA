import pandas as pd
import numpy as np
import glob
import os
import argparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

def calculate_metrics(merged_df, gt_metric_col, pred_col, direction='higher_is_better'):
    """
    Calculates Pearson, Spearman, Loss, AUROC for a single target-model pair.
    """
    if len(merged_df) < 2:
        return np.nan, np.nan, np.nan, np.nan

    y_true = merged_df[gt_metric_col].values
    y_pred = merged_df[pred_col].values

    # Pearson / Spearman
    # Handle constant input (std=0) to avoid warnings
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        p_corr = 0.0
        s_corr = 0.0
    else:
        p_corr, _ = pearsonr(y_true, y_pred)
        s_corr, _ = spearmanr(y_true, y_pred)

    # Loss (Regret)
    # Loss = |Best_True - True_of_Best_Pred|
    # Find index of best prediction
    if direction == 'higher_is_better': # e.g. TM-Score
        best_pred_idx = np.argmax(y_pred)
        best_true_val = np.max(y_true)
        selected_true_val = y_true[best_pred_idx]
        loss = best_true_val - selected_true_val
    else: # e.g. RMSD (Lower is better)
        best_pred_idx = np.argmin(y_pred) # Predictor thinks this is lowest (best)
        best_true_val = np.min(y_true)    # Actually lowest
        selected_true_val = y_true[best_pred_idx] # What we actually got
        loss = selected_true_val - best_true_val 

    # AUROC (Top 25 classification)
    # Define "Positive" class as being in the Top 25% of GROUND TRUTH
    if direction == 'higher_is_better':
        threshold = np.percentile(y_true, 75)
        y_class = (y_true >= threshold).astype(int)
    else:
        threshold = np.percentile(y_true, 25) # Top 25% is lowest 25% values
        y_class = (y_true <= threshold).astype(int)
        
    # Check if only one class present
    if len(np.unique(y_class)) < 2:
        auroc = 0.5 # Neutral
    else:
        # For AUROC, y_score needs to be correlated with Positive Class.
        # If higher_is_better: Higher Pred should mean Positive (High True). OK.
        # If lower_is_better: Lower Pred should mean Positive (Low True).
        # roc_auc_score expects Higher Score -> Class 1.
        # So if lower_is_better, we must flip y_pred signs or use 1/x for AUC calc
        if direction == 'lower_is_better':
             auroc = roc_auc_score(y_class, -y_pred)
        else:
             auroc = roc_auc_score(y_class, y_pred)
             
    return p_corr, s_corr, loss, auroc

def main():
    parser = argparse.ArgumentParser(description="Calculate Metrics from Raw Predictions (with 80% Rule)")
    parser.add_argument("--pred_dir", required=True, help="Directory containing raw prediction CSVs (e.g. T1104.csv)")
    parser.add_argument("--score_dir", required=True, help="Directory containing Ground Truth Score CSVs")
    parser.add_argument("--output", required=True, help="Output CSV filename")
    
    # We need to map which column in Pred matches which column in Score
    # Simple assumption: User provides pairs
    parser.add_argument("--metric_pairs", required=True, nargs='+', help="Pairs of PredCol:TruthCol (e.g. tm_score:tm_score)")
    parser.add_argument("--directions", required=True, nargs='+', help="Directions (higher_is_better/lower_is_better) matching the pairs")

    args = parser.parse_args()
    
    # Parse metrics
    metrics_map = []
    for item in args.metric_pairs:
        p, t = item.split(':')
        metrics_map.append((p, t))
        
    if len(metrics_map) != len(args.directions):
        print("Error: Metric pairs and directions count mismatch")
        return

    pred_files = glob.glob(os.path.join(args.pred_dir, "*.csv"))
    results = []
    
    print(f"Processing {len(pred_files)} prediction files...")
    
    for pred_idx, f in enumerate(pred_files):
        target = os.path.basename(f).replace('.csv', '')
        
        # Load Predictions
        try:
            pred_df = pd.read_csv(f)
        except:
            continue
            
        # Identify Models (groups) in this prediction file?
        # Usually 'model' column contains 'MULTICOM_1', 'AlphaFold_1' etc.
        # We need to extract the Group Name.
        # ASSUMPTION: 'model' column exists. 'Group' is substring? 
        # Or does the USER input merged predictions where 'Model' is the Group?
        # Let's assume standard format: 'model' column is 'DecoyName'. 
        # WAIT. A raw prediction file usually comes from ONE group.
        # But here the user presumably has 'oof_fold_1.csv' etc which has 'RF_tmscore' vs 'GroundTruth'.
        # OR the user has 'CASP16_community_dataset' which has files per target?
        # Let's assume we are grading typical CASP submissions where we might have multiple groups per target 
        # OR a merged dataframe.
        # Let's assume the user uses this for THEIR predictions usually.
        # But provided data is likely: Target, Model(Decoy), GroupName?
        # If the input CSV has 'Model' as Decoy, how do we know the Predictor Group?
        # Maybe the file IS the Predictor's submission?
        pass

        # Let's change strategy: The user has separate files per Target.
        # Inside each file, there are predictions for many decoys.
        # Who made these predictions? 
        # If it's the "Community Dataset", maybe it has all groups?
        # Let's assume the file contains columns like 'Group1_score', 'Group2_score'?
        # OR the input directory is structured as `Predictions/Group/Target.csv`?
        
        # Given the ambiguity, let's implement a robust "Group" detector or assume the file contains
        # columns for each predictor (like in the EMA dataframe we established earlier).
        # Check 'oof_fold_1.csv' format from memory.
        # It has `target`, `model` (decoy), and many columns `RF_tmscore`, `Stacking_lddt` (Predictors).
        # This is a Perfect Format! One file per fold, but we can iterate.
        # SCRIPT INPUT: Assume Single CSV or Directory of CSVs where columns are predictors.
        
        pass 
    
    # RE-PLAN for script arguments:
    # Option A: Directory of Targets, each file has decoys. Columns are Predictors.
    # Option B: Single Merged CSV (Target, Decoy, Pred1, Pred2, ...)
    
    # Let's support Option A (Directory) as it fits "pred_dir".
    
    for f in pred_files:
        target = os.path.basename(f).replace('.csv', '')
        
        # Load Ground Truth
        score_file = os.path.join(args.score_dir, f"{target}_quality_scores.csv")
        if not os.path.exists(score_file):
            print(f"Ground truth missing for {target}, skipping.")
            continue
            
        try:
            gt_df = pd.read_csv(score_file)
            pred_df = pd.read_csv(f)
        except:
            continue
            
        # Merge GT and Preds to align decoys
        # Keys: 'model' in pred, 'model_name' in gt (usually)
        # We need to standardize keys.
        
        # Prepare GT key
        if 'model_name' in gt_df.columns:
            gt_df['key'] = gt_df['model_name'].astype(str).str.replace('.pdb', '').str.strip()
        elif 'model' in gt_df.columns:
             gt_df['key'] = gt_df['model'].astype(str).str.strip()
        else:
            continue
            
        # Prepare Pred key
        if 'model' in pred_df.columns:
            pred_df['key'] = pred_df['model'].astype(str).str.strip()
        else:
            continue
            
        # 1. Total Decoys in Target (from GT)
        # We consider 'valid' GT entries those that have non-NaN values for the metrics we care about.
        # But to keep it simple, let's count rows in GT.
        N_total = len(gt_df)
        if N_total == 0: continue
        
        merged = pd.merge(pred_df, gt_df, on='key', suffixes=('_pred', '_true'))
        
        # Identify Predictor Columns
        # Assuming all columns except 'key', 'model', 'target' are predictors?
        # Or user specifies predictor columns?
        # Better: Assume all columns in pred_df that are numeric are predictors, excluding key/model/target.
        metadata_cols = ['key', 'model', 'target', 'Target', 'Model', 'merge_key']
        predictor_cols = [c for c in pred_df.columns if c not in metadata_cols and pd.api.types.is_numeric_dtype(pred_df[c])]
        
        for predictor in predictor_cols:
            # Per Predictor Logic
            
            # Check 80% Rule
            # Count non-NaN predictions for this predictor matches
            # We must look at 'merged', because we only care about decoys that HAVE a ground truth.
            # (If we predicted a decoy that doesn't exist in GT, it doesn't help evaluation)
            
            # Predictor's specific column in merged
            p_col = predictor
            
            # Filter where Predictor has value AND GT has value (for each metric pair)
            for (p_metric_name, t_metric_col), direction in zip(metrics_map, args.directions):
                # Wait, 'predictor' variable IS the score column?
                # Example: predictor='RF_tmscore'.
                # We need to know which GT metric it predicts.
                # Complex case: We might have RF_tmscore, RF_lddt in same file.
                # Argument --metric_pairs is tricky if columns are named arbitrarily.
                
                # Simplified Assumption:
                # The user extracts specific columns or we rely on substring matching?
                # "RF_tmscore" predicts "tm_score".
                # "Stacking_lddt" predicts "lddt".
                
                # Let's simplify: This script grades ONE metric type at a time?
                # Or assume predictor_cols IS the score?
                # No, mixed metrics in one file is common.
                
                # Alternative: User passes ONE Predictor Column and ONE Truth Column? 
                # No, too slow.
                
                # Let's try heuristic:
                # User provides: --gt_metric tm_score --gt_metric lddt
                # Script finds all columns in pred_df.
                # User provides map as args?
                
                # Let's fallback to specific use case: User wants to grade THEIR models.
                # They likely have a CSV: target, model, MyMethod_TM, MyMethod_LDDT.
                # They want to map MyMethod_TM -> tm_score (GT).
                # MyMethod_LDDT -> lddt (GT).
                
                # If we iterate over ALL numeric columns, we need to know the target metric for each.
                # Let's just output logic for ALL numeric columns against ALL gt columns? No.
                
                # Solution: Loop over --metric_pairs.
                # For `pred_col_pattern:gt_col`.
                # identifying cols that match `pred_col_pattern` (e.g. `*_tmscore` matches `tm_score`).
                pass

                # Actually, simpler: just match the metric name suffix?
                # Let's iterate metrics defined in args.
                # p_suffix = p_metric_name (e.g. "tmscore")
                # t_col = t_metric_col (e.g. "tm_score")
                
                # Find all columns in pred_df ending with p_suffix?
                # e.g. "RF_tmscore" ends with "tmscore"?
                # or just contains it?
                pass
                
                # To be robust, let's just make the user do the mapping externaly/renaming 
                # OR support the exact structure of `oof_fold_*.csv`.
                # `oof` cols: `RF_tmscore_mmalign`, `Stacking_lddt`...
                # GT cols: `tm_score`, `lddt`...
                
                # Let's perform exact matching if possible, or simple contains.
                # We will check if `predictor` column seems to target `t_metric_col`.
                
                is_match = False
                # 1. Exact Name match?
                if predictor == p_metric_name: is_match = True
                # 2. Predictor contains metric name?
                elif p_metric_name in predictor: is_match = True
                
                if not is_match: continue
                
                # Found a pair: predict `predictor` vs truth `t_metric_col`
                
                # Subset valid data
                valid_mask = merged[predictor].notna() & merged[t_metric_col].notna()
                valid_data = merged[valid_mask]
                
                n_valid = len(valid_data)
                coverage = n_valid / N_total
                
                model_name = predictor # This is the "Method Name"
                
                row = {
                    'Target': target,
                    'Model': model_name,
                    'Metric_Type': p_metric_name, # e.g. 'tmscore'
                    'N_Total': N_total,
                    'N_Pred': n_valid,
                    'Coverage': coverage
                }
                
                if coverage < 0.8:
                    # FAIL 80% Rule
                    row['Pearson'] = np.nan
                    row['Spearman'] = np.nan
                    row['Loss'] = np.nan
                    row['AUROC'] = np.nan
                else:
                    # PASS
                    p, s, l, a = calculate_metrics(valid_data, t_metric_col, predictor, direction)
                    row['Pearson'] = p
                    row['Spearman'] = s
                    row['Loss'] = l
                    row['AUROC'] = a
                
                results.append(row)

    if results:
        res_df = pd.DataFrame(results)
        res_df.to_csv(args.output, index=False)
        print(f"Grading Complete. Saved to {args.output}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
