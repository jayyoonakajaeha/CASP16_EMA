import pandas as pd
import numpy as np
import argparse
import sys

def calculate_z_scores_2pass(df, target_col, model_col, metric_col, direction='higher_is_better', threshold=2.0):
    """
    Calculates Z-scores using the CASP16 2-Pass Outlier Removal algorithm.
    
    1. Pass 1: Calculate Mean/Std. Filter out 'poor' models (> 2 STD worse than mean).
       - If 'higher_is_better': Remove score < Mean - 2*Std
       - If 'lower_is_better': Remove score > Mean + 2*Std
    2. Pass 2: Recalculate Mean/Std from remaining models.
    3. Calculate Z-scores for ALL models using standard parameters from Pass 2.
    4. Clamp negative Z-scores to 0.0.
    """
    
    # Ensure directional handling for Loss/RMSD
    if direction == 'lower_is_better':
        # Internal processing: Flip sign for standard processing logic?
        # Or standard logic: Z = (Mean - Score) / Std  (so lower score gives positive Z)
        # CASP method: Z = (d_i - Mean) / Std?? No.
        # Let's stick to standard definition: Z = (x - mu) / sigma
        # And if lower is better (e.g. RMSD), we want low RMSD to have High Z-score.
        # So we invert the metric inputs just for calculation?
        # CASP16 global_analysis code did: z_loss = (-1) * z_loss
        # So we calculate standard Z-score first, then flip sign?
        # Wait, for OUTLIER detection, we must be careful.
        # Bad RMSD is High. So outlier is > Mean + 2*Std.
        pass
        
    scores = df[metric_col].values
    
    # 1. First Pass Statistics
    # Filter valid scores (remove NaNs)
    valid_scores = scores[~np.isnan(scores)]
    
    if len(valid_scores) < 2:
        return np.zeros(len(scores))
        
    mu1 = np.mean(valid_scores)
    sigma1 = np.std(valid_scores, ddof=1)
    
    if sigma1 == 0:
        return np.zeros(len(scores))
    
    # Identify outliers (Models that are too bad)
    if direction == 'higher_is_better':
        # Bad = Low. Outlier < Mean - 2*Std
        cutoff = mu1 - (threshold * sigma1)
        mask_keep = valid_scores >= cutoff
    else: # lower_is_better (e.g. RMSD)
        # Bad = High. Outlier > Mean + 2*Std
        cutoff = mu1 + (threshold * sigma1)
        mask_keep = valid_scores <= cutoff
        
    filtered_scores = valid_scores[mask_keep]
    
    # 2. Second Pass Statistics
    if len(filtered_scores) < 2:
        # If filtering removes almost everything, fall back to Pass 1 stats OR return 0?
        # CASP usually keeps Pass 1 if too few models remain, but let's assume robust.
        mu2 = np.mean(filtered_scores)
        sigma2 = np.std(filtered_scores, ddof=1)
    else:
        mu2 = np.mean(filtered_scores)
        sigma2 = np.std(filtered_scores, ddof=1)
        
    if sigma2 == 0:
        sigma2 = sigma1 # Fallback
        if sigma2 == 0: return np.zeros(len(scores))

    # 3. Calculate Z-scores
    # Z = (x - mu) / sigma
    z_scores = (scores - mu2) / sigma2
    
    # If lower_is_better, flip sign so that Lower Score (Better) -> Higher Z
    if direction == 'lower_is_better':
        z_scores = z_scores * -1.0
        
    # 4. Clamping
    # Provide 0.0 for models that are worse than the reference mean (negative Z)
    z_scores = np.where(z_scores < 0, 0.0, z_scores)
    
    # Handle NaNs (missing predictions get 0.0 penalty implicitly by FillNM or explicit handling)
    # The input DF might have NaNs.
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    
    return z_scores

def main():
    parser = argparse.ArgumentParser(description="CASP16 Z-Score Evaluation (2-Pass Algorithm)")
    parser.add_argument("--input", required=True, help="Input CSV file with Target, Model, and Metric columns")
    parser.add_argument("--output", required=True, help="Output CSV file for Leaderboard")
    parser.add_argument("--metrics", required=True, nargs='+', help="List of metrics to evaluate (e.g. tm_score lddt)")
    parser.add_argument("--directions", required=True, nargs='+', help="Directions for metrics (higher_is_better or lower_is_better). Must match order of metrics.")
    parser.add_argument("--threshold", type=float, default=2.0, help="Z-score threshold for outlier removal (Default: 2.0)")
    
    args = parser.parse_args()
    
    if len(args.metrics) != len(args.directions):
        print("Error: Number of metrics and directions must match.")
        sys.exit(1)
        
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Ensure required columns
    required_cols = ['Target', 'Model'] + args.metrics
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' missing from input CSV.")
            sys.exit(1)
            
    # Process each metric
    ranking_dfs = []
    
    unique_targets = df['Target'].unique()
    print(f"Found {len(unique_targets)} targets.")
    
    grand_sums = {} # Model -> Total Z-Score Sum across all metrics?
    # Usually we evaluate each metric separately first.
    
    overall_results = []
    
    for metric, direction in zip(args.metrics, args.directions):
        print(f"Processing {metric} ({direction})...")
        
        # We need to calculate Z-scores PER TARGET
        # Add a temporary Z column
        z_col_name = f"Z_{metric}"
        df[z_col_name] = 0.0
        
        for target in unique_targets:
            # Slice for this target
            t_mask = df['Target'] == target
            t_df = df[t_mask].copy()
            
            # Identify Model column and Metric values
            # Handle duplicates? Assume unique Model per Target
            
            z_vals = calculate_z_scores_2pass(t_df, 'Target', 'Model', metric, direction, args.threshold)
            
            # Assign back
            df.loc[t_mask, z_col_name] = z_vals
            
        # Sum Z-scores per Model (Ranking Score for this metric)
        leaderboard = df.groupby('Model')[z_col_name].sum().reset_index()
        leaderboard.columns = ['Model', 'Summed_Z_Score']
        leaderboard = leaderboard.sort_values(by='Summed_Z_Score', ascending=False)
        leaderboard['Rank'] = range(1, len(leaderboard) + 1)
        leaderboard['Metric'] = metric
        
        overall_results.append(leaderboard)
        
        # Save individual metric leaderboard
        leaderboard.to_csv(f"{args.output}_{metric}.csv", index=False)
        print(f"Saved leaderboard for {metric} to {args.output}_{metric}.csv")

    # Combine all results if needed?
    # For now, separate metric leaderboards are standard.
    
if __name__ == "__main__":
    main()
