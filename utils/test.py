import pandas as pd
from models.model import GapfillModel, Transformer_GapfillModel, GapfillModel_NoEmbed, Transformer_NoEmbed,GapfillModel_NoRevin
import os
import torch
from torch.utils.data import DataLoader
from data.dataset import test_Gapfill_Dataset
import pickle
from tqdm import tqdm  # Import progress bar library
import numpy as np
from models.CNN_LSTM import CNN_LSTM


def test_gap_filler ( args, site_information, site_name_to_id, site_igbp_to_id, unique_sites_num, unique_igbp_num,scalers):
    source_to_folder = {
        "FLUXNET": "FLX",
        "AmeriFLUX": "AMF",
        "ICOS": "ICOS"
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load model

    model_path = os.path.join(args.output_path, args.filling_var, f'exp_{args.exp_num}')
    loss_suffix = "" if args.enable_QC_loss else "_withoutQCLoss"
    best_model_path = os.path.join(model_path,
                                   f'{args.model_name}_{args.filling_var}_best_model_{args.seq_len}_exp{args.exp_num}'
                                   f'{loss_suffix}.pth')
    # Initialize model and move to specified device (CPU/GPU)
    if args.model_name == "MOETransformer":
        model = GapfillModel(args, unique_sites_num, unique_igbp_num, 'test').to(device)
    elif args.model_name == "MOETransformer_NoEmbed":
        model = GapfillModel_NoEmbed(args, "test").to(device)
    elif args.model_name == "Transformer":
        model = Transformer_GapfillModel(args, unique_sites_num, unique_igbp_num, 'test').to(device)
    elif args.model_name == 'Transformer_NoEmbed':
        model = Transformer_NoEmbed(args, 'test').to(device)
    elif args.model_name == 'MOETransformer_NoRevin':
        model = GapfillModel_NoRevin(args, unique_sites_num, unique_igbp_num, 'train').to(device)
    elif args.model_name == 'CNN_LSTM':
        model = CNN_LSTM(args).to(device)
    else:
        raise ValueError(f'model not define ')
    # Load model parameters
    if os.path.exists(best_model_path):
        # Load with map_location to ensure model is on the correct device
        model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=device))
        model.eval()  # Set to evaluation mode
        print(f"Successfully loaded model parameters from {best_model_path}, running on {device}")
    else:
        raise FileNotFoundError(f"Model file not found: {best_model_path}")
    # Test code, test each site separately
    for index, row, in site_information.iterrows():
        site_pred = []
        # If normalization is performed, load the normalization parameters of the training set
        site_name = row['SITE_ID']
        source = row['SOURCE']
        test_dataset = test_Gapfill_Dataset(args, row, site_igbp_to_id, site_name_to_id,scalers)
        # Generate test data
        data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
        # Test model
        test_bar = tqdm(data_loader, desc=f' {site_name}_{args.filling_var} [Test]', leave=False)
        with torch.no_grad():
            for model_input, targets in test_bar:
                for key in model_input:
                    model_input[key] = model_input[key].to(device)
                outputs = model(model_input)
                outputs = test_dataset.denormalize(tensor=outputs,var_type='obs')
                site_pred.extend(outputs.cpu().numpy().tolist())
        # Save prediction results to npy file
        site_pred = np.array(site_pred)
        site_pred_path = os.path.join(args.output_path, args.filling_var, f'exp_{args.exp_num}',
                                      source_to_folder[source], site_name)
        os.makedirs(site_pred_path, exist_ok=True)
        site_pred_name = f'{site_name}_{args.filling_var}_pred_{args.model_name}_{args.seq_len}_exp_{args.exp_num}{loss_suffix}.npy'
        site_pred_file = os.path.join(site_pred_path, site_pred_name)
        np.save(site_pred_file, site_pred)
        print(f"Prediction results for {args.filling_var} of site {site_name} have been saved to {site_pred_file}")
    print("Test completed")
