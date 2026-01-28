import os.path
import numpy as np
from .load_utils import find_fluxnet_files
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

source_to_folder = {
    "FLUXNET": "FLX",
    "AmeriFLUX": "AMF",
    "ICOS": "ICOS"
}


def pad_and_build_sequences ( data_array, seq_len, pad_value ):
    """
    在数组的开头进行填充，然后构建滑动窗口序列。
    Pads the beginning of an array and then builds sliding window sequences.

    Args:
        data_array (np.ndarray): The original numpy array. It can be 1D or 2D.
                                 原始的Numpy数组，可以是一维或二维。
        seq_len (int): The final length of each sequence.
                       每个序列的最终长度。
        pad_value (int or float): The value to use for padding.
                                  用于填充的值。

    Returns:
        np.ndarray: A 3D array of sequences with shape (len(data_array), seq_len, num_features).
                    一个形状为 (原始数组长度, 序列长度, 特征数) 的三维序列数组。
    """
    # --- 步骤 1: 准备和填充 ---

    # Ensure the input array is at least 2D for consistent processing
    if data_array.ndim == 1:
        data_array = data_array.reshape(-1, 1)
    num_features = data_array.shape[1]
    num_pads = seq_len
    # Create the padding array only if necessary
    if num_pads > 0:
        padding_array = np.full((num_pads, num_features), pad_value)
        padded_array = np.vstack([padding_array, data_array])
    else:
        padded_array = data_array
    # --- 步骤 2: 构建时间序列 ---
    # The number of sequences will be equal to the length of the original data array
    num_samples = len(data_array)
    # Use a sliding window on the padded array to create the sequences
    sequences = np.array([padded_array[i: i + seq_len] for i in range(num_samples)])
    return sequences


def set_rg_tag ( df, rg_flag ):
    df["rg_rank"] = np.select(
        condlist=[
            df[rg_flag] < 10,
            (df[rg_flag] > 10) & (df[rg_flag] < 100),
            df[rg_flag] > 100
        ],
        choicelist=[
            1,
            2,
            3
        ],
        default=0
    )
    return df, ["rg_rank"]


def set_season_tag ( df, isnorth=True ):
    if isnorth:
        df["season"] = (df.index.month % 12 + 3) // 3  # print(seasons)
    else:
        df["season"] = ((df.index.month + 6) % 12 + 3) // 3
    return df, ["season"]


def get_quality_mask ( args, obs_df, filling_var ):
    """
    根据观测数据(obs_df)中的QC列是否存在，生成质量标签(quality flag)。

    Generates quality flags based on the existence of a QC column in the observation dataframe.

    Args:
        args (Namespace): 包含配置的对象，此函数会修改其中的 default_miss_flag。
        obs_df (pd.DataFrame): 包含观测数据的DataFrame。
        filling_var (str): 需要处理的目标变量名。

    Returns:
        pd.Series: 生成的质量标签序列。
    """
    qc_var = f"{filling_var}_QC"
    # 逻辑分支1: 如果QC变量存在
    if qc_var in obs_df.columns:
        print(f"    -> 检测到QC列 '{qc_var}'，将基于QC值生成质量标签。")
        # Set the default missing flag for padding, as requested
        # 按要求设置用于填充的默认缺失标签
        args.default_miss_flag = 4
        args.is_QC = True
        # Copy the QC column and fill any of its own missing values with the default flag
        # 复制QC列，并将其自身的缺失值（NaN）填充为默认标签4
        quality = obs_df[qc_var].copy().fillna(args.default_miss_flag)
    # 逻辑分支2: 如果QC变量不存在
    else:
        print(f"    -> 未检测到QC列，将基于 '{filling_var}' 列的数值是否存在来生成质量标签。")
        # Set the default missing flag for padding, as requested
        args.default_miss_flag = 1
        args.is_QC = False
        quality = obs_df[filling_var].isna().astype(int)
    return quality.astype(int).to_numpy()


def parse_timestamp ( ts ):
    try:
        return pd.to_datetime(ts, format='%Y%m%d%H%M')
    except ValueError:
        try:
            return pd.to_datetime(ts)
        except ValueError:
            return pd.NaT


def filter_driver_data ( args, driver_data, time_range, seq_len, time_resolution ):
    """
    Filter driver_data based on time range and sequence length.

    Parameters:
        driver_data (pd.DataFrame): DataFrame containing timestamp and other driver variables.
        time_range (pd.Series): Time range containing valid time points.
        seq_len (int): Sequence length, i.e., the number of data points before each valid time point.
        time_resolution (pd.Timedelta): Time resolution, i.e., the time interval of each time step.

    Returns:
        pd.DataFrame: Filtered driver_data.
    """
    # Ensure driver_data is sorted by timestamp
    driver_data = driver_data.sort_values(by='TIMESTAMP_END').reset_index(drop=True)
    driver_data['TIMESTAMP_END'] = parse_timestamp(driver_data['TIMESTAMP_END'])
    time_range = pd.to_datetime(time_range, format='%Y%m%d%H%M')
    # Calculate extended time range
    min_time = time_range.min() - seq_len * time_resolution
    max_time = time_range.max()
    # Filter driver_data
    filtered_driver_data = driver_data[
        (driver_data['TIMESTAMP_END'] >= min_time) & (driver_data['TIMESTAMP_END'] <= max_time)
        ]
    return filtered_driver_data


def build_driver_data ( driver_data, seq_len, aux_variables ):
    """
    Build driver dataset based on all time points and their seq_len data points before, only including specified auxiliary variables.

    Parameters:
        driver_data (pd.DataFrame): DataFrame containing timestamp and other driver variables.
        seq_len (int): Sequence length, i.e., the number of data points before each time point.
        aux_variables (list): List of auxiliary variables to build.

    Returns:
        np.ndarray: Built driver dataset.
    """
    # Ensure driver_data is sorted by timestamp
    driver_data = driver_data.sort_values(by='TIMESTAMP_END').reset_index(drop=True)

    # Convert DataFrame to numpy array for faster processing
    driver_data_np = driver_data[aux_variables].values

    # Calculate the number of sequences
    num_sequences = len(driver_data_np) - seq_len

    # Initialize the result array
    driver_sequences = np.zeros((num_sequences, seq_len, len(aux_variables)), dtype=driver_data_np.dtype)

    # Build sequences using numpy slicing
    for i in range(num_sequences):
        driver_sequences[i] = driver_data_np[i:i + seq_len]

    return driver_sequences


def normalize_data ( data, normalization_type='min_max', scaler=None ):
    """
    Normalize ndarray data, supports 2D and 3D data.

    Parameters:
        data (np.ndarray): Data to be normalized, shape can be:
                          - 2D: (n_samples, n_features)
                          - 3D: (n_samples, seq_len, n_features)
        normalization_type (str): Normalization type, optional 'min_max' or 'z_score'
        scaler (object|list): If provided, use pre-trained normalizer
                             - For 3D data, should be a scaler list of length n_features
        feature_axis (int): Axis of features (usually 2 or -1 for 3D data)

    Returns:
        np.ndarray: Normalized data (same shape as input)
        object|list: Normalizer (if no scaler is provided, returns trained normalizer or scaler list)
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy ndarray")

    original_ndim = data.ndim

    # Handle 3D data
    if original_ndim == 3:
        # Reshape 3D data to 2D (n_samples*seq_len, n_features) for normalization
        n_samples, seq_len, n_features = data.shape
        data_2d = data.reshape(-1, n_features)
        # If scaler list is provided
        if scaler is not None and isinstance(scaler, list):
            if len(scaler) != n_features:
                raise ValueError(
                    f"Provided scaler list length ({len(scaler)}) doesn't match feature number ({n_features})")
            # Apply corresponding scaler to each feature
            normalized_data = np.zeros_like(data_2d)
            for i in range(n_features):
                normalized_data[:, i] = scaler[i].transform(data_2d[:, i:i + 1]).flatten()
            normalized_data = normalized_data.reshape(n_samples, seq_len, n_features)
        else:
            # Normalize entire 2D data
            normalized_data_2d, scaler = normalize_data(data_2d, normalization_type, scaler)
            # Reshape result back to 3D
            normalized_data = normalized_data_2d.reshape(n_samples, seq_len, n_features)
            # If new scaler is created, convert it to a scaler list for each feature
            if scaler is not None and not isinstance(scaler, list):
                # For global scaler, create an independent scaler for each feature
                scaler_list = []
                for i in range(n_features):
                    if normalization_type == 'min_max':
                        new_scaler = MinMaxScaler()
                        new_scaler.min_ = scaler.min_[i]
                        new_scaler.scale_ = scaler.scale_[i]
                        new_scaler.data_min_ = scaler.data_min_[i]
                        new_scaler.data_max_ = scaler.data_max_[i]
                    elif normalization_type == 'z_score':
                        new_scaler = StandardScaler()
                        new_scaler.mean_ = scaler.mean_[i]
                        new_scaler.scale_ = scaler.scale_[i]
                        new_scaler.var_ = scaler.var_[i]
                    else:
                        raise ValueError('Not define normalized type')
                    scaler_list.append(new_scaler)
                scaler = scaler_list

        return normalized_data, scaler
    # Original logic for handling 2D data
    if normalization_type == 'min_max':
        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(data)
        normalized_data = scaler.transform(data)
        print('Normalization using min_max')

    elif normalization_type == 'z_score':
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(data)
        normalized_data = scaler.transform(data)
        print('Normalization using z_score')
    else:
        raise ValueError("Invalid normalization type. Choose from 'min_max' or 'z_score'.")
    return normalized_data, scaler


def denormalize_data ( data, scaler, normalization_type=None ):
    """
    Denormalize data, supports 2D and 3D data.

    Parameters:
        data (np.ndarray): Data to be denormalized
        scaler (object|list): Scaler used during normalization or scaler list
        normalization_type (str): Normalization type (if scaler is list, must provide)

    Returns:
        np.ndarray: Denormalized data
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy ndarray")

    # Handle 3D data
    if data.ndim == 3:
        n_samples, seq_len, n_features = data.shape
        data_2d = data.reshape(-1, n_features)

        if isinstance(scaler, list):
            if len(scaler) != n_features:
                raise ValueError(f"Scaler list length ({len(scaler)}) doesn't match feature number ({n_features})")

            denormalized_data = np.zeros_like(data_2d)
            for i in range(n_features):
                denormalized_data[:, i] = scaler[i].inverse_transform(data_2d[:, i:i + 1]).flatten()
        else:
            if normalization_type is None:
                raise ValueError("For global scaler with 3D data, normalization_type must be provided")

            if normalization_type == 'min_max':
                temp_scaler = MinMaxScaler()
            else:
                temp_scaler = StandardScaler()

            # Use global scaler parameters
            denormalized_data = temp_scaler.inverse_transform(data_2d)

        return denormalized_data.reshape(n_samples, seq_len, n_features)

    # Handle 2D data
    return scaler.inverse_transform(data)


def generate_data ( args, site_name_to_id, igbp_to_id, select_site):
    # Define a dictionary to map the source names to folder names
    # Initialize lists to store all data
    all_train_observation = []
    all_train_driver_history = []
    all_train_driver_current = []
    all_train_night_mask = []
    all_train_quality = []
    all_train_input_value = []
    all_train_valid_mask = []
    all_val_observation = []
    all_val_driver_history = []
    all_val_driver_current = []
    all_val_night_mask = []
    all_val_quality = []
    all_val_input_value = []
    all_val_valid_mask = []
    # Iterate through each row in the select_site dataframe
    for index, row in select_site.iterrows():
        # Get the site information
        site_name = row['SITE_ID'].strip()
        source = row['SOURCE'].strip()
        igbp = row['IGBP'].strip()
        resolution = row['TIME_RESOLUTION']
        # Get file paths
        suffix = '_debias' if args.data_type == 'debias' else ''
        file_path = os.path.join(args.data_path, source_to_folder[source], site_name)
        observation_file_path = find_fluxnet_files(file_path, source_to_folder[source])
        driver_file_path = os.path.join(file_path, f'{site_name}_Era5land_{resolution}{suffix}.csv')
        driver_cols_to_read = ['TIMESTAMP_START', 'TIMESTAMP_END'] + args.aux_variables
        print(f"    -> 准备从驱动文件中加载 {len(driver_cols_to_read)} 列。")
        driver_data = pd.read_csv(
            driver_file_path,
            usecols=driver_cols_to_read,
            parse_dates=['TIMESTAMP_END']
        )
        # Load data
        observation_data = pd.read_csv(observation_file_path, na_values=[-9999])
        # Generate or load valid mask
        valid_mask_path = os.path.join(args.output_path, args.filling_var, source_to_folder[source], site_name)
        valid_mask_file = os.path.join(valid_mask_path, f"{site_name}_{args.filling_var}_valid_mask.npy")
        quanlity_file = os.path.join(valid_mask_path, f"{site_name}_{args.filling_var}_quality_{args.seq_len}.npy")
        path_to_check = [valid_mask_file, quanlity_file]
        if not all(os.path.exists(p) for p in path_to_check):
            valid_mask = get_quality_mask(args, observation_data, args.filling_var)
            os.makedirs(valid_mask_path, exist_ok=True)
            np.save(os.path.join(valid_mask_path, f"{site_name}_{args.filling_var}_valid_mask.npy"),
                    valid_mask)
            quality_data = pad_and_build_sequences(valid_mask, args.seq_len, args.default_miss_flag)
            np.save(quanlity_file, quality_data)
            print(f'{site_name} valid mask generated')
        else:
            print(f'{site_name} valid mask already exists')
            valid_mask = np.load(valid_mask_file)
            quality_data = np.load(quanlity_file)
        night_mask_path = os.path.join(args.output_path, 'night', source_to_folder[source], site_name)
        night_mask_file = os.path.join(night_mask_path, f"{site_name}_night_mask.npy")
        if not os.path.exists(night_mask_file):
            if 'NIGHT' in observation_data.columns:
                night_mask = observation_data['NIGHT'].copy().fillna(2).astype(int).to_numpy()
                os.makedirs(night_mask_path, exist_ok=True)
                np.save(night_mask_file, night_mask)
            else:
                raise ValueError(f"NIGHT not exist in {site_name}")
        else:
            night_mask = np.load(night_mask_file)
        # Save site all observation data
        all_observation_data_path = os.path.join(args.output_path, args.filling_var, source_to_folder[source],
                                                 site_name)
        all_observation_data_file = os.path.join(all_observation_data_path, f'{site_name}_{args.filling_var}_obs.npy')
        all_input_value_file = os.path.join(all_observation_data_path,
                                            f'{site_name}_{args.filling_var}_input_value_{args.seq_len}.npy')
        obs_path = [all_observation_data_file, all_input_value_file]
        if not all(os.path.exists(p) for p in obs_path):
            all_observation_data = observation_data[args.filling_var].fillna(0).values.reshape(-1, 1)
            os.makedirs(all_observation_data_path, exist_ok=True)
            np.save(all_observation_data_file, all_observation_data)
            input_value = pad_and_build_sequences(all_observation_data, args.seq_len, 0)
            np.save(all_input_value_file, input_value)
        else:
            all_observation_data = np.load(all_observation_data_file)
            input_value = np.load(all_input_value_file)
        # Process driver data
        all_driver_file_path = os.path.join(args.output_path, 'driver', source_to_folder[source], site_name)
        all_driver_history_file = os.path.join(all_driver_file_path,
                                               f'{site_name}_all_driver_history_{args.seq_len}.npy')
        all_driver_current_file = os.path.join(all_driver_file_path,
                                               f'{site_name}_all_driver_current.npy')
        driver_path = [all_driver_history_file, all_driver_current_file]
        if not all(os.path.exists(p) for p in driver_path):
            print(f'Generating driver data of {site_name}')
            feature_columns = args.aux_variables + ['tod', 'doy', 'site_id', 'igbp']
            time_range = observation_data['TIMESTAMP_END']
            time_resolution = pd.Timedelta(hours=0.5) if resolution == 'HH' else pd.Timedelta(hours=1)
            driver_data = filter_driver_data(args, driver_data, time_range, args.seq_len, time_resolution)
            driver_data['TIMESTAMP_END'] = pd.to_datetime(driver_data['TIMESTAMP_END'], format='%Y%m%d%H%M')
            driver_data['site_id'] = site_name_to_id[site_name]
            driver_data['igbp'] = igbp_to_id[igbp]
            driver_data['tod'] = driver_data['TIMESTAMP_END'].dt.hour * 2 + (
                    driver_data['TIMESTAMP_END'].dt.minute // 30)
            driver_data['doy'] = driver_data['TIMESTAMP_END'].dt.dayofyear - 1
            all_driver_history = build_driver_data(driver_data, args.seq_len,
                                                   feature_columns)
            all_driver_history_data = np.stack(all_driver_history, axis=0)
            all_driver_current_data = driver_data.iloc[args.seq_len:][feature_columns].to_numpy()
            os.makedirs(all_driver_file_path, exist_ok=True)
            np.save(all_driver_history_file, all_driver_history_data)
            np.save(all_driver_current_file, all_driver_current_data)
        else:
            all_driver_history_data = np.load(all_driver_history_file)
            all_driver_current_data = np.load(all_driver_current_file)
        # Split data into train and validation (80% train, 20% validation)
        n_samples = len(all_driver_current_data)
        train_size = int(n_samples * args.train_ratio)
        all_shapes = [
            all_observation_data.shape[0],
            all_driver_history_data.shape[0],
            all_driver_current_data.shape[0],
            night_mask.shape[0],
            quality_data.shape[0],
            input_value.shape[0],
            valid_mask.shape[0]
        ]
        # 2. Use the property of a set to check if all values are identical.
        #    If all numbers in the list are the same, the length of the set will be 1.
        if len(set(all_shapes)) > 1:
            # If the set's length is greater than 1, it means there's at least one different value.
            raise ValueError(
                f"数据数组的第一维度（样本数）不匹配！请检查数据准备流程。"
                f"检测到的样本数: {all_shapes}"
            )
        print(" -> 断言通过：所有数据数组的样本数一致。")
        if args.shuffle:
            # Shuffle the data
            shuffled_indices = np.random.permutation(n_samples)
            all_observation_data = all_observation_data[shuffled_indices]
            all_driver_history_data = all_driver_history_data[shuffled_indices]
            all_driver_current_data = all_driver_current_data[shuffled_indices]
            night_mask = night_mask[shuffled_indices]
            quality_data = quality_data[shuffled_indices]
            input_value = input_value[shuffled_indices]
            valid_mask = valid_mask[shuffled_indices]
        # Split the data
        train_obs = all_observation_data[:train_size]
        val_obs = all_observation_data[train_size:]
        train_driver_history = all_driver_history_data[:train_size]
        val_driver_history = all_driver_history_data[train_size:]
        train_driver_current = all_driver_current_data[:train_size]
        val_driver_current = all_driver_current_data[train_size:]
        train_night_mask = night_mask[:train_size]
        val_night_mask = night_mask[train_size:]
        train_quality = quality_data[:train_size]
        val_quality = quality_data[train_size:]
        train_input_value = input_value[:train_size]
        val_input_value = input_value[train_size:]
        train_valid_mask = valid_mask[:train_size]
        val_valid_mask = valid_mask[train_size:]
        # Append to lists
        all_train_observation.append(train_obs)
        all_train_driver_history.append(train_driver_history)
        all_train_driver_current.append(train_driver_current)
        all_train_night_mask.append(train_night_mask)
        all_train_input_value.append(train_input_value)
        all_train_quality.append(train_quality)
        all_train_valid_mask.append(train_valid_mask)
        all_val_observation.append(val_obs)
        all_val_driver_history.append(val_driver_history)
        all_val_driver_current.append(val_driver_current)
        all_val_night_mask.append(val_night_mask)
        all_val_input_value.append(val_input_value)
        all_val_quality.append(val_quality)
        all_val_valid_mask.append(val_valid_mask)
        print(f"{site_name} data processed (train: {len(train_obs)}, val: {len(val_obs)})")
    # Concatenate all sites' data
    all_train_observation = np.concatenate(all_train_observation, axis=0) if len(all_train_observation) > 1 else \
        all_train_observation[0]
    all_train_driver_history = np.concatenate(all_train_driver_history, axis=0) if len(
        all_train_driver_history) > 1 else all_train_driver_history[0]
    all_train_driver_current = np.concatenate(all_train_driver_current, axis=0) if len(
        all_train_driver_current) > 1 else all_train_driver_current[0]
    all_train_night_mask = np.concatenate(all_train_night_mask, axis=0) if len(all_train_night_mask) > 1 else \
        all_train_night_mask[0]
    all_train_input_value = np.concatenate(all_train_input_value, axis=0) if len(all_train_input_value) > 1 else \
        all_train_input_value[0]
    all_train_quality = np.concatenate(all_train_quality, axis=0) if len(all_train_quality) > 1 else \
        all_train_quality[0]
    all_train_valid_mask = np.concatenate(all_train_valid_mask, axis=0) if len(all_train_valid_mask) > 1 else \
        all_train_valid_mask[0]
    all_val_observation = np.concatenate(all_val_observation, axis=0) if len(all_val_observation) > 1 else \
        all_val_observation[0]
    all_val_driver_history = np.concatenate(all_val_driver_history, axis=0) if len(all_val_driver_history) > 1 else \
    all_val_driver_history[0]
    all_val_driver_current = np.concatenate(all_val_driver_current, axis=0) if len(all_val_driver_current) > 1 else \
    all_val_driver_current[0]
    all_val_night_mask = np.concatenate(all_val_night_mask, axis=0) if len(all_val_night_mask) > 1 else \
        all_val_night_mask[0]
    all_val_input_value = np.concatenate(all_val_input_value, axis=0) if len(all_val_input_value) > 1 else \
        all_val_input_value[0]
    all_val_quality = np.concatenate(all_val_quality, axis=0) if len(all_val_quality) > 1 else \
        all_val_quality[0]
    all_val_valid_mask = np.concatenate(all_val_valid_mask, axis=0) if len(all_val_quality) > 1 else \
        all_val_valid_mask[0]
    # Create output directories if they don't exist
    train_output_path = os.path.join(args.output_path, args.filling_var)
    val_output_path = os.path.join(args.output_path, args.filling_var)
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)
    # Save the concatenated data
    np.save(os.path.join(train_output_path, f"train_all_observation_{args.filling_var}.npy"), all_train_observation)
    np.save(os.path.join(train_output_path, f"train_all_driver_history_{args.seq_len}.npy"), all_train_driver_history)
    np.save(os.path.join(train_output_path, f"train_all_driver_current.npy"), all_train_driver_current)
    np.save(os.path.join(train_output_path, f"train_all_night_mask_{args.filling_var}.npy"), all_train_night_mask)
    np.save(os.path.join(train_output_path, f"train_all_quality_{args.filling_var}_{args.seq_len}.npy"),
            all_train_quality)
    np.save(os.path.join(train_output_path, f"train_all_input_value_{args.filling_var}_{args.seq_len}.npy"),
            all_train_input_value)
    np.save(os.path.join(train_output_path, f'train_all_valid_mask.npy'), all_train_valid_mask)
    np.save(os.path.join(val_output_path, f"val_all_observation_{args.filling_var}.npy"), all_val_observation)
    np.save(os.path.join(val_output_path, f"val_all_driver_history_{args.seq_len}.npy"), all_val_driver_history)
    np.save(os.path.join(val_output_path, f"val_all_driver_current.npy"), all_val_driver_current)
    np.save(os.path.join(val_output_path, f"val_all_night_mask_{args.filling_var}.npy"), all_val_night_mask)
    np.save(os.path.join(val_output_path, f"val_all_quality_{args.filling_var}_{args.seq_len}.npy"), all_val_quality)
    np.save(os.path.join(val_output_path, f"val_all_input_value_{args.filling_var}_{args.seq_len}.npy"),
            all_val_input_value)
    np.save(os.path.join(val_output_path, f'val_all_valid_mask.npy'), all_val_valid_mask)
    return all_train_observation, all_train_input_value, all_train_driver_history, all_train_driver_current, all_train_night_mask, all_train_quality, all_train_valid_mask


def generate_test_data ( args, select_site, site_igbp_to_id, site_name_to_id ):
    # Define a dictionary to map the source names to folder names
    # Get the site information
    site_name = select_site['SITE_ID'].strip()
    source = select_site['SOURCE'].strip()
    igbp = select_site['IGBP'].strip()
    resolution = select_site['TIME_RESOLUTION']
    # Get file paths
    suffix = '_debias' if args.data_type == 'debias' else ''
    file_path = os.path.join(args.data_path, source_to_folder[source], site_name)
    observation_file_path = find_fluxnet_files(file_path, source_to_folder[source])
    driver_file_path = os.path.join(file_path, f'{site_name}_Era5land_{resolution}{suffix}.csv')
    driver_cols_to_read = ['TIMESTAMP_START', 'TIMESTAMP_END'] + args.aux_variables
    print(f"    -> 准备从驱动文件中加载 {len(driver_cols_to_read)} 列。")
    driver_data = pd.read_csv(
        driver_file_path,
        usecols=driver_cols_to_read,  # <-- 使用 usecols 参数指定读取的列
        parse_dates=['TIMESTAMP_END']
    )

    # Load data
    observation_data = pd.read_csv(observation_file_path, na_values=[-9999])
    # Generate or load valid mask
    valid_mask_path = os.path.join(args.output_path, args.filling_var, source_to_folder[source], site_name)
    valid_mask_file = os.path.join(valid_mask_path, f"{site_name}_{args.filling_var}_valid_mask_obs.npy")
    quanlity_file = os.path.join(valid_mask_path, f"{site_name}_{args.filling_var}_quality_{args.seq_len}.npy")
    path_to_check = [valid_mask_file, quanlity_file]
    if not all(os.path.exists(p) for p in path_to_check):
        valid_mask = get_quality_mask(args, observation_data, args.filling_var)
        os.makedirs(valid_mask_path, exist_ok=True)
        np.save(os.path.join(valid_mask_path, f"{site_name}_{args.filling_var}_valid_mask.npy"),
                valid_mask)
        quality_data = pad_and_build_sequences(valid_mask, args.seq_len, args.default_miss_flag)
        np.save(quanlity_file, quality_data, )
        print(f'{site_name} valid mask generated')
    else:
        print(f'{site_name} valid mask already exists')
        valid_mask = np.load(valid_mask_file)
        quality_data = np.load(quanlity_file)
    night_mask_path = os.path.join(args.output_path, 'night', source_to_folder[source], site_name)
    night_mask_file = os.path.join(night_mask_path, f"{site_name}_night_mask.npy")
    if not os.path.exists(night_mask_file):
        if 'NIGHT' in observation_data.columns:
            night_mask = observation_data['NIGHT'].copy().fillna(2).astype(int).to_numpy()
            os.makedirs(night_mask_path, exist_ok=True)
            np.save(night_mask_file, night_mask)
        else:
            raise ValueError(f"NIGHT not exist in {site_name}")
    else:
        night_mask = np.load(night_mask_file)
    # Save site all observation data
    all_observation_data_path = os.path.join(args.output_path, args.filling_var, source_to_folder[source],
                                             site_name)
    all_observation_data_file = os.path.join(all_observation_data_path, f'{site_name}_{args.filling_var}_obs.npy')
    all_input_value_file = os.path.join(all_observation_data_path,
                                        f'{site_name}_{args.filling_var}_input_value_{args.seq_len}.npy')
    obs_path = [all_observation_data_file, all_input_value_file]
    if not all(os.path.exists(p) for p in obs_path):
        all_observation_data = observation_data[args.filling_var].fillna(0).values.reshape(-1, 1)
        os.makedirs(all_observation_data_path, exist_ok=True)
        np.save(all_observation_data_file, all_observation_data)
        input_value = pad_and_build_sequences(all_observation_data, args.seq_len, 0)
        np.save(all_input_value_file, input_value)
    else:
        all_observation_data = np.load(all_observation_data_file)
        input_value = np.load(all_input_value_file)
    # Process driver data
    all_driver_file_path = os.path.join(args.output_path, 'driver', source_to_folder[source], site_name)
    all_driver_history_file = os.path.join(all_driver_file_path, f'{site_name}_all_driver_history_{args.seq_len}.npy')
    all_driver_current_file = os.path.join(all_driver_file_path, f'{site_name}_all_driver_current.npy')
    driver_path = [all_driver_history_file,all_driver_current_file]
    if not all(os.path.exists(p) for p in driver_path):
        print(f'Generating driver data of {site_name}')
        feature_columns = args.aux_variables + ['tod', 'doy', 'site_id', 'igbp']
        time_range = observation_data['TIMESTAMP_END']
        time_resolution = pd.Timedelta(hours=0.5) if resolution == 'HH' else pd.Timedelta(hours=1)
        driver_data = filter_driver_data(args, driver_data, time_range, args.seq_len, time_resolution)
        driver_data['TIMESTAMP_END'] = pd.to_datetime(driver_data['TIMESTAMP_END'], format='%Y%m%d%H%M')
        driver_data['site_id'] = site_name_to_id[site_name]
        driver_data['igbp'] = site_igbp_to_id[igbp]
        driver_data['tod'] = driver_data['TIMESTAMP_END'].dt.hour * 2 + (
                driver_data['TIMESTAMP_END'].dt.minute // 30)
        driver_data['doy'] = driver_data['TIMESTAMP_END'].dt.dayofyear - 1
        all_driver_seq = build_driver_data(driver_data, args.seq_len,
                                           feature_columns)
        all_driver_history_data = np.stack(all_driver_seq, axis=0)
        all_driver_current_data = driver_data.iloc[args.seq_len:][feature_columns].to_numpy()
        os.makedirs(all_driver_file_path, exist_ok=True)
        np.save(all_driver_history_file, all_driver_history_data)
        np.save(all_driver_current_file,all_driver_current_data)
    else:
        all_driver_history_data = np.load(all_driver_history_file)
        all_driver_current_data = np.load(all_driver_current_file)
    all_shapes = [
        all_observation_data.shape[0],
        all_driver_history_data.shape[0],
        all_driver_current_data.shape[0],
        night_mask.shape[0],
        quality_data.shape[0],
        input_value.shape[0],
        valid_mask.shape[0]
    ]
    # 2. Use the property of a set to check if all values are identical.
    #    If all numbers in the list are the same, the length of the set will be 1.
    if len(set(all_shapes)) > 1:
        # If the set's length is greater than 1, it means there's at least one different value.
        raise ValueError(
            f"数据数组的第一维度（样本数）不匹配！请检查数据准备流程。"
            f"检测到的样本数: {all_shapes}"
        )
    print(" -> 断言通过：所有数据数组的样本数一致。")
    return all_observation_data,input_value,all_driver_history_data,all_driver_current_data, night_mask, quality_data, valid_mask
