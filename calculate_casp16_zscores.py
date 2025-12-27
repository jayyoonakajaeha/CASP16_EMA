import pandas as pd
import numpy as np
import argparse
import sys
import os

def calculate_z_vector_2pass(values, direction='higher_is_better', threshold=2.0):
    """
    Calculates Z-scores for a 1D vector using 2-Pass Outlier Removal.
    
    1. Pass 1: standard Z-score. Remove outliers > threshold (default 2.0).
    2. Pass 2: Re-calculate Mean/Std on inliers.
    3. Compute Z for ALL original values using Pass 2 stats.
    4. Flip sign if 'lower_is_better' so that positive Z is good.
    5. Clamp negative Z to 0.0.
    """
    # Filter Nans
    # In CASP16 extraction, we assume we have values.
    # But input might have NaNs ? (e.g. from 80% rule failure -> but 80% rule fails the whole target)
    # If values has NaN, we propagate NaN or 0? 
    # Usually we ignore NaNs for Mean/Std calc.
    
    valid_mask = ~np.isnan(values)
    valid_vals = values[valid_mask]
    
    if len(valid_vals) < 2:
        return np.zeros(len(values))
        
    # Pass 1
    mu1 = np.mean(valid_vals)
    sigma1 = np.std(valid_vals, ddof=1)
    
    if sigma1 == 0:
        return np.zeros(len(values))
        
    if direction == 'higher_is_better':
        # Bad is Low. Outlier < Mean - k*Std
        cutoff = mu1 - (threshold * sigma1)
        inlier_mask = valid_vals >= cutoff
    else: # lower_is_better (e.g. Loss)
        # Bad is High. Outlier > Mean + k*Std
        cutoff = mu1 + (threshold * sigma1)
        inlier_mask = valid_vals <= cutoff
        
    inliers = valid_vals[inlier_mask]
    
    # Pass 2
    if len(inliers) < 2:
        # Fallback to Pass 1 if too aggressive
        mu2, sigma2 = mu1, sigma1
    else:
        mu2 = np.mean(inliers)
        sigma2 = np.std(inliers, ddof=1)
        
    if sigma2 == 0:
        sigma2 = sigma1
        if sigma2 == 0: return np.zeros(len(values))
        
    # Final Z-score for ALL values
    z_scores = (values - mu2) / sigma2
    
    # Directionality Flip
    if direction == 'lower_is_better':
        z_scores = z_scores * -1.0
        
    # Clamp Negative Z to 0
    # Note: NaNs in z_scores (from NaNs in values) remain NaNs or become 0?
    # CASP: "Missing" -> 0.0 penalty.
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    z_scores = np.where(z_scores < 0, 0.0, z_scores)
    
    return z_scores

def process_metric_group(df, group_name, metrics, use_loss=True, threshold=2.0):
    """
    Processes a group of metrics (e.g. SCORE = [tm_score])
    Returns a Series of summed RS scores per Model.
    """
    # Filter for relevant metrics
    group_df = df[df['Metric_Type'].isin(metrics)].copy()
    
    if group_df.empty:
        print(f"Warning: No data found for metrics {metrics} in group {group_name}")
        return None

    # We need to calculate RS per Target per Metric
    # RS = 0.5*Z(P) + 0.5*Z(S) + Z(A) + Z(L)
    
    # Unique Targets
    targets = group_df['Target'].unique()
    
    # We will accumulate RS per Model
    model_rs_sum = {}
    
    print(f"--- Processing {group_name} ({len(metrics)} metrics) ---")
    
    for metric in metrics:
        subset = group_df[group_df['Metric_Type'] == metric].copy()
        if subset.empty: continue
        
        # We process separately per target
        for target in subset['Target'].unique():
            t_df = subset[subset['Target'] == target].copy()
            
            # Calculate Z-scores for each component
            # Pearson: Higher is Better
            zp = calculate_z_vector_2pass(t_df['Pearson'].values, 'higher_is_better', threshold)
            
            # Spearman: Higher is Better
            zs = calculate_z_vector_2pass(t_df['Spearman'].values, 'higher_is_better', threshold)
            
            # AUROC: Higher is Better
            # (Note: grade_predictions handles direction logic for AUC input)
            za = calculate_z_vector_2pass(t_df['AUROC'].values, 'higher_is_better', threshold)
            
            # Loss: Lower is Better
            # grade_predictions outputs Loss > 0 (Min difference). Lower is better.
            zl = calculate_z_vector_2pass(t_df['Loss'].values, 'lower_is_better', threshold)
            
            # Formula
            if use_loss:
                rs_values = (0.5 * zp) + (0.5 * zs) + za + zl
            else:
                rs_values = (0.5 * zp) + (0.5 * zs) + za
                
            # Add to accumulators
            models = t_df['Model'].values
            for m, rs in zip(models, rs_values):
                model_rs_sum[m] = model_rs_sum.get(m, 0.0) + rs
                
    # Convert to DataFrame
    res = pd.DataFrame(list(model_rs_sum.items()), columns=['Model', f'RS_{group_name}'])
    res = res.sort_values(by=f'RS_{group_name}', ascending=False)
    
    # Add Rank
    res['Rank'] = range(1, len(res) + 1)
    return res

def main():
    parser = argparse.ArgumentParser(description="Calculate CASP16 Ranking Scores (RS) from Graded Metrics")
    parser.add_argument("--input", required=True, help="Input CSV (Output of grade_predictions.py)")
    parser.add_argument("--output_dir", required=True, help="Directory to save leaderboards")
    
    # Metric Groups
    parser.add_argument("--tm_metric", nargs='+', default=[], help="Metrics for SCORE (e.g. tm_score)")
    parser.add_argument("--qs_metric", nargs='+', default=[], help="Metrics for QSCORE (e.g. qs_best dockq_wave)")
    parser.add_argument("--local_metric", nargs='+', default=[], help="Metrics for Local Quality (Loss excluded)")
    parser.add_argument("--threshold", type=float, default=2.0, help="Z-score outlier threshold (Default: 2.0)")

    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Process SCORE (Global Topology)
    if args.tm_metric:
        print("Calculating SCORE (Global Topology)...")
        score_df = process_metric_group(df, "SCORE", args.tm_metric, use_loss=True, threshold=args.threshold)
        if score_df is not None:
            path = f"{args.output_dir}/leaderboard_SCORE.csv"
            score_df.to_csv(path, index=False)
            print(f"Saved {path}")
            
    # Process QSCORE (Interface Accuracy)
    if args.qs_metric:
        print("Calculating QSCORE (Interface Accuracy)...")
        qs_df = process_metric_group(df, "QSCORE", args.qs_metric, use_loss=True, threshold=args.threshold)
        if qs_df is not None:
            path = f"{args.output_dir}/leaderboard_QSCORE.csv"
            qs_df.to_csv(path, index=False)
            print(f"Saved {path}")
            
    # Process Local Quality (No Loss)
    if args.local_metric:
        print("Calculating Local Quality (No Loss)...")
        local_df = process_metric_group(df, "Local", args.local_metric, use_loss=False, threshold=args.threshold)
        if local_df is not None:
            path = f"{args.output_dir}/leaderboard_Local.csv"
            local_df.to_csv(path, index=False)
            print(f"Saved {path}")

if __name__ == "__main__":
    main()
