import argparse


def get_config ():
    """
    Configuration setup for flux data gap-filling process.
    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Configuration for flux data gap-filling')
    # data config
    auxiliary_variables_set = [
        "2m_temperature_C",
        "soil_temperature_level_1_C",
        "VPD",
        "RH",
        "surface_pressure",
        "wind speed",
        'total_precipitation',
        'surface_latent_heat_flux',
        'surface_sensible_heat_flux',
        'surface_net_radiation',
        'surface_solar_radiation_downwards',
        'surface_thermal_radiation_downwards',
        'leaf_area_index_high_vegetation',
        'leaf_area_index_low_vegetation',
        'volumetric_soil_water_layer_1'
    ]
    # train config
    parser.add_argument('--exp_num', type=str, default='1')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--enable_QC_loss', type=bool, default=True)
    # data config
    parser.add_argument('--analyze_site_file', type=str, default='./data/config/select_site_UTC.xlsx')
    parser.add_argument('--default_mean', type=float, default=None)
    parser.add_argument("--default_std", type=float, default=None)
    parser.add_argument('--is_QC', type=bool, default=True)
    parser.add_argument('--miss_ratio', type=int)
    parser.add_argument("--default_miss_flag", type=int, default=4)
    parser.add_argument("--data_type", type=str, default='debias')
    parser.add_argument('--max_missing_percentage', type=float, default=80)
    parser.add_argument('--data_path', type=str, default='E:/data_new')  # E:/data or /home/tianxu/data
    parser.add_argument('--output_path', type=str, default='E:/output')  # E:/output or /home/tianxu/output
    parser.add_argument('--aux_variables', type=list, default=auxiliary_variables_set)
    parser.add_argument('--Gen_flag', type=bool, default=False)
    parser.add_argument('--global_normalize', type=bool, default=False)
    parser.add_argument('--site_info_path', type=str, default='./data/config/test_select_site_UTC.xlsx')
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--loss', type=str, default='HUBER')  # MAE ,MSE ,HUBER,
    parser.add_argument('--shuffle', type=bool, default=False)
    # filling config
    # Variable to fill
    parser.add_argument('--filling_var', type=str, default='G_F_MDS', help='Variable to fill. Options: G_F_MDS,'
                                                                           'H_F_MDS, LE_F_MDS,'
                                                                           'LW_IN_F_MDS, PA '
                                                                           'TA_F_MDS,'
                                                                           'NETRAD, VPD_F_MDS, WS, NEE_VUT_REF')
    # model config
    parser.add_argument('--model_name', type=str,
                        default='MOETransformer')  # MOETransformer,MOETransformer_NoEmbed,Transformer,
    # MOETransformer_NoRevin
    # Transformer_NoEmbed,,CNN_LSTM
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=1)
    parser.add_argument('--feature_in', type=int, default=len(auxiliary_variables_set))
    parser.add_argument('--n_layers', type=int, default=1)
    # Embed_dim * 4 = hidden_dim
    # moe config
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--shared_expert_num', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.1)
    # hyper parameter
    # if filling_var == P tweedie_p=1.4671 filling_var == SW_IN_MDS tweedie_p = 1.5830
    parser.add_argument("--tweedie_p", type=float, default=1.5)
    parser.add_argument('--delta', type=int, default=1, help='huber loss delta')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--learning_rate_for_loss', type=float, default=0.001)
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_config()
    print(config)
