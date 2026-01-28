import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from  config.cfg import get_config
from utils.load_utils import load_site_information


def calculate_metrics(obs, pred, night_mask, valid_mask, min_samples=10):
    """
    Calculates various performance metrics (RMSE, R2, Bias, NSE, KGE) for observed vs. predicted values,
    split by overall, night, and day periods.

    Args:
        obs (np.array): Array of observed values.
        pred (np.array): Array of predicted values.
        night_mask (np.array): Boolean mask indicating night time periods.
        valid_mask (np.array): Boolean mask indicating valid data points.
        min_samples (int): Minimum number of samples required to compute metrics.

    Returns:
        dict: A dictionary containing metric results for 'overall', 'night', and 'day' periods.
    """

    def compute_all_metrics ( y_true, y_pred ):
        """
        Helper function to compute all metrics for a given set of true and predicted values.
        """
        n = len(y_true)
        # 初始化默认的失败返回字典
        nan_result = {
            "RMSE": np.nan,
            "R2": np.nan,
            "Bias": np.nan,
            "KGE": np.nan,
            "Comment": ""
        }

        if n < min_samples:
            nan_result["Comment"] = f"Too few samples (n={n})"
            return nan_result

        # 移除这个多余的 check，或者如果保留，也要返回字典
        if len(y_true) < 2:
            nan_result["Comment"] = "Less than 2 samples"
            return nan_result

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        bias = mean_absolute_error(y_true, y_pred)

        # Kling-Gupta Efficiency (KGE)
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)

        if mean_true == 0 or std_true == 0 or mean_pred == 0:
            nan_result["Comment"] = "Zero mean or std, cannot compute KGE"
            nan_result["RMSE"] = rmse
            nan_result["R2"] = r2
            nan_result["Bias"] = bias
            nan_result["KGE"] = np.nan
            return nan_result

        r = np.corrcoef(y_pred, y_true)[0, 1] if len(y_pred) > 1 and len(y_true) > 1 else np.nan
        beta = mean_pred / mean_true
        gamma = (std_pred / mean_pred) / (std_true / mean_true)

        kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) if all(
            np.isfinite([r, beta, gamma])) else np.nan

        return {
            "RMSE": rmse,
            "R2": r2,
            "Bias": bias,
            "KGE": kge,
            "Comment": ""
        }

    # Apply valid mask to observations and predictions
    obs_valid = obs[valid_mask==0].flatten()
    pred_valid = pred[valid_mask==0].flatten()
    night_mask_valid = night_mask[valid_mask==0].flatten()

    # Separate data into night and day periods
    obs_night = obs_valid[night_mask_valid==1]
    pred_night = pred_valid[night_mask_valid==1]

    obs_day = obs_valid[night_mask_valid==0]
    pred_day = pred_valid[night_mask_valid==0]

    # Compute metrics for overall, night, and day periods
    return {
        "overall": compute_all_metrics(obs_valid, pred_valid),
        "night": compute_all_metrics(obs_night, pred_night),
        "day": compute_all_metrics(obs_day, pred_day)
    }


def evaluate(args,site_information=None):
    """
    Evaluates the model performance for each site in the provided site information.

    Args:
        args (argparse.Namespace): Command-line arguments.
        site_information (pd.DataFrame): DataFrame containing site information.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each site.
    """
    source_to_folder = {
        "FLUXNET": "FLX",
        "AmeriFLUX": "AMF",
        "ICOS": "ICOS"
    }
    loss_suffix = "" if args.enable_QC_loss else "_withoutQCLoss"
    all_site_metric = pd.DataFrame()
    if site_information is not None:
        print(f"Site information loaded successfully. Total sites: {len(site_information)}.")
        for index, row in site_information.iterrows():
            site_name = row['SITE_ID']
            source = row['SOURCE']
            IGBP = row['IGBP']
            print(f"Processing site: {site_name} from source {source}")
            try:
                base_path = os.path.join(args.output_path, args.filling_var, source_to_folder[source], site_name)
                # Load valid mask
                valid_mask_path = os.path.join(base_path, f'{site_name}_{args.filling_var}_valid_mask.npy')
                if not os.path.exists(valid_mask_path):
                    raise FileNotFoundError(f"Valid mask file not found for site {site_name}.")
                valid_mask = np.load(valid_mask_path)
                # Load night mask
                night_mask_path = os.path.join(args.output_path, "night", source_to_folder[source], site_name,
                                               f"{site_name}_night_mask.npy")
                if not os.path.exists(night_mask_path):
                    raise FileNotFoundError(f"Night mask file not found for site {site_name}.")
                night_mask = np.load(night_mask_path)
                # Load observed and predicted data
                obs_path = os.path.join(base_path, f'{site_name}_{args.filling_var}_obs.npy')
                # pre_path
                pred_path = os.path.join(args.output_path, args.filling_var,f'exp_{args.exp_num}',source_to_folder[source], site_name)
                pred_path = os.path.join(pred_path, f'{site_name}_{args.filling_var}_pred_{args.model_name}_{args.seq_len}_exp_{args.exp_num}{loss_suffix}.npy')
                if not os.path.exists(obs_path) or not os.path.exists(pred_path):
                    raise FileNotFoundError(f"Obs or pred file not found for site {site_name}.")
                obs = np.load(obs_path)
                pred = np.load(pred_path)
                # Ensure all masks and data arrays have the same length
                if not (len(obs) == len(pred) == len(valid_mask) == len(night_mask)):
                    raise ValueError(f"Data length mismatch for site {site_name}. Obs: {len(obs)}, Pred: {len(pred)}, Valid Mask: {len(valid_mask)}, Night Mask: {len(night_mask)}")
                # Calculate metrics using the dedicated function
                metrics = calculate_metrics(obs, pred, night_mask, valid_mask)
                # Flatten metrics into a row for the DataFrame
                metrics_flat = {
                    'SITE_ID': site_name,
                    'SOURCE': source,
                    'IGBP': IGBP,
                }
                for period in ['overall', 'night', 'day']:
                    for metric in ['RMSE', 'R2', 'Bias','KGE']:
                        metrics_flat[f'{metric}_{period}'] = metrics[period][metric]
                    metrics_flat[f'Comment_{period}'] = metrics[period]['Comment']

                all_site_metric = pd.concat([all_site_metric, pd.DataFrame([metrics_flat])], ignore_index=True)
                print(f'The {site_name} metrics have been calculated and added to the summary.')
            except Exception as e:
                print(f"Error processing site {site_name}: {e}")
        # Save all site results

        output_file = os.path.join(args.output_path, args.filling_var,f"exp_{args.exp_num}", f"{args.filling_var}_metrics_summary_{args.seq_len}_exp{args.exp_num}_{args.model_name}{loss_suffix}.csv")
        all_site_metric.to_csv(output_file, index=False)
        print(f"All metrics saved to {output_file}")
    else:
        print("No site information loaded. Please check 'test_select_site.xls'.")