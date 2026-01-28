import os.path
import pandas as pd
from config.cfg import get_config
from utils.load_utils import load_site_information, filter_sites
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
from data.dataset import Gapfill_Train_Dataset, Gapfill_Val_Dataset
from utils.train import train_gap_filler
from utils.test import test_gap_filler
import pickle
from evaluate import evaluate
import os


def set_random_seed ( seed ):
    """
        seed (int): random seed
    """
    # set NumPy random seed
    np.random.seed(seed)
    # set PyTorch random seed
    torch.manual_seed(seed)
    # if you use CUDA，set cuda random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main ( args ):
    # Set the random seed
    set_random_seed(args.random_seed)
    # Print a message to indicate the filling variable and filling flag
    print(f"Now we prepare to fill the missing data in column {args.filling_var}")
    # Load the site information
    site_information = load_site_information(args.site_info_path)
    # Check if the site information was loaded successfully
    if site_information is not None:
        # get unique site names and IGBP
        site_names = site_information['SITE_ID'].unique().tolist()
        site_igbp = site_information['IGBP'].unique().tolist()
        # Create dictionaries to map site names and IGBP to unique IDs
        site_name_to_id = {name: i for i, name in enumerate(site_names)}
        igbp_to_id = {igbp_val: i for i, igbp_val in enumerate(site_igbp)}
        unique_sites_num = len(site_names)  # 或 len(site_name_to_id)
        unique_igbp_num = len(site_igbp)  # 或 len(igbp_to_id)
        print(f"Site information loaded successfully. Total sites: {len(site_information)}.")
        # Print a message to indicate that we are starting to filter the sites based on missing percentage
        print('Now we stat to filter the site based on missing percentage')
        select_site_file_name = os.path.join(args.output_path, args.filling_var,
                                             f'select_site_{args.filling_var}_max_missing_percentage{args.max_missing_percentage}.xlsx')
        # Filter the sites based on the filling variable and filling flag
        filtered_sites = filter_sites(site_information, args.filling_var,
                                      args.max_missing_percentage)
        # Save the filtered sites to an Excel file
        os.makedirs(os.path.join(args.output_path, args.filling_var), exist_ok=True)
        filtered_sites.to_excel(select_site_file_name, index=False)
        # Print a message to indicate the number of sites filtered
        if len(filtered_sites) > 0:
            print(f"Filtered sites based on {args.filling_var} missing percentage. Total sites: {len(filtered_sites)}.")
        else:
            print("No sites were filtered based on missing percentage.")
    else:
        # Print a message to indicate that the site information failed to load
        print("Site information loading failed.")
    # Check if the train flag is set
    scalers = {}
    gap_fill_train_dataset = Gapfill_Train_Dataset(args, site_name_to_id, igbp_to_id, filtered_sites)
    if not args.global_normalize:
        # 这部分如果进行全局标准化的话会出现问题，只有在使用Hybrid Revin时才计算全局标准化参数
        global_mean, global_std = gap_fill_train_dataset.calculate_global_stats()
        args.default_mean = global_mean
        args.default_std = global_std
    scalers = gap_fill_train_dataset.scalers
    gap_fill_val_dataset = Gapfill_Val_Dataset(args, scalers=scalers)
    if args.is_train:
        # Create a SummaryWriter object to log information
        log_dir = os.path.join("./logs", f"{args.model_name}_{args.filling_var}")
        writer = SummaryWriter(log_dir=log_dir)
        train_gap_filler(args, gap_fill_train_dataset, gap_fill_val_dataset, unique_sites_num, unique_igbp_num, writer)
    do_test = True
    if do_test:
        test_gap_filler(args, site_information, site_name_to_id, igbp_to_id, unique_sites_num, unique_igbp_num, scalers)
    evaluate(args, site_information)

if __name__ == "__main__":
    args = get_config()
    main(args)
