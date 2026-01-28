import calendar
import os.path
import glob
from datetime import date, timedelta
import pandas as pd
import numpy as np
import os
import xarray as xr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


def calculate_metrics ( y_true, y_pred ):
    """
    计算评估指标：MSE, R, R²
    :param y_true: 真实值
    :param y_pred: 预测值
    :return: 评估指标字典
    """
    # 确保输入是 NumPy 数组
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # 调整 y_true 的维度为 (-1, 1)
    y_true = y_true.reshape(-1, 1)
    # 计算评估指标
    mse = mean_squared_error(y_true, y_pred)
    r = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {'MSE': mse, 'R': r, 'R^2': r2, 'MAE': mae}


def write_results_to_csv ( results, output_csv_path ):
    """
    将结果写入CSV文件
    :param results: 包含站点名称和评估指标的列表
    :param output_csv_path: 输出CSV文件路径
    """
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)


def extract_data_with_mask ( args, csv_file, variable_name ):
    """
    从CSV文件中提取指定变量的数据，并根据值是否等于-9999创建掩码，
    将值等于-9999的部分赋值为NaN。

    参数:
        csv_file (str): CSV文件的路径。
        variable_name (str): 要提取的变量名称。

    返回:
        data_masked (pd.Series): 经过掩码处理的数据，值为-9999的部分为NaN。
        mask (pd.Series): 掩码，True表示值不等于-9999，False表示值等于-9999。
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise IOError(f"无法读取CSV文件 '{csv_file}'：{e}")
    # 检查变量列是否存在
    if variable_name not in df.columns:
        raise ValueError(f"变量 '{variable_name}' 不存在于CSV文件中。")
    # 提取数据

    df = df.set_index(df["TIMESTAMP_START"])
    # 创建掩码：值不等于-9999
    mask = df[variable_name] != -9999
    # 设置数据为NaN，值等于-9999的部分
    label = df[variable_name].replace(-9999, np.nan)
    return label, mask


def extract_data_with_mask_QC ( args, csv_file, variable_name ):
    """
    从CSV文件中提取指定变量的数据，并根据变量名_QC列为0的部分创建掩码，将非0的部分赋值为NaN。
    参数:
        csv_file (str): CSV文件的路径。
        variable_name (str): 要提取的变量名称。

    返回:
        data_masked (pd.Series): 经过掩码处理的数据，非0的QC部分为NaN。
        mask (pd.Series): 掩码，True表示QC为0，False表示QC不为0。
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        raise IOError(f"无法读取CSV文件 '{csv_file}': {e}")
    # 检查变量列是否存在
    if variable_name not in df.columns:
        raise ValueError(f"变量 '{variable_name}' 不存在于CSV文件中。")
    # 构造QC列名
    qc_column = f"{variable_name}_QC"
    # 检查QC列是否存在
    if qc_column not in df.columns:
        raise ValueError(f"QC变量列 '{qc_column}' 不存在于CSV文件中。")
    # 设置索引
    df = df.set_index(df["TIMESTAMP_START"])
    label = df[variable_name].copy()
    label_qc = df[qc_column].copy()
    # 创建掩码：QC == 0
    mask = label_qc == 0
    # 将mask为False的部分对应的label值赋值为NaN
    label[~mask] = pd.NA
    return label, mask


def extract_year_month ( timestamp ):
    """
    从时间戳中提取年份和月份。
    参数:
    timestamp (str): 时间戳字符串，格式为 'YYYYMMDDHHMM'
    返回:
    tuple: 包含年份和月份的元组 (year, month)
    """
    # 提取年份和月份
    timestamp = str(timestamp)
    year_and_month = timestamp[0:6]
    year = str(year_and_month[:4])
    month = str(year_and_month[4:])
    return year, month


def generate_date_combinations ( start_time, end_time ):
    start_year, start_month = start_time.year, start_time.month
    end_year, end_month = end_time.year, end_time.month
    start_year, start_month, end_year, end_month = int(start_year), int(start_month), int(end_year), int(end_month)
    start_date = date(start_year, start_month, 1)
    end_date = date(end_year, end_month, 1)
    if start_date > end_date:
        return []
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        _, num_days = calendar.monthrange(current_date.year, current_date.month)

        last_day_of_month = date(current_date.year, current_date.month, num_days)

        date_list.append((current_date.year, current_date.month))

        current_date = last_day_of_month + timedelta(days=1)
    return date_list


def get_label ( args, site_name, site_source ):
    """
    从指定数据源获取标签数据

    参数:
        args: 配置参数
        site_name (str): 站点名称
        site_source (str): 数据源类型 ('FLUXNET', 'AmeriFlux', 'ICOS')
    """
    file_folder = os.path.join(args.data_path, site_name)
    if not os.path.exists(file_folder):
        raise FileNotFoundError(f'文件夹 {file_folder} 不存在。')
    # 根据数据源类型设置文件前缀
    prefix_map = {
        'FLUXNET': 'FLX',
        'AmeriFLUX': 'AMF',
        'ICOS': 'ICOSETC'
    }
    if site_source not in prefix_map:
        raise ValueError(f"不支持的数据源类型: {site_source}。支持的类型为: {', '.join(prefix_map.keys())}")

    prefix = prefix_map[site_source]
    label_file = glob.glob(os.path.join(file_folder, f'{prefix}*.csv'))
    if len(label_file) == 0:
        raise FileNotFoundError(f"在文件夹 {file_folder} 中未找到以 {prefix} 开头的csv文件")
    elif len(label_file) > 1:
        raise ValueError(f"在文件夹 {file_folder} 中找到多个以 {prefix} 开头的csv文件")
    label_name = label_file[0]
    if args.is_QC_var:
        label, mask = extract_data_with_mask_QC(args, label_name, args.filling_var)
    else:
        label, mask = extract_data_with_mask(args, label_name, args.filling_var)
    return label, mask


def get_aux_data ( args, source, site_name, timestamp_start, timestamp_end ):
    """
    提取辅助变量数据，从多个 .nc 文件中读取指定变量，拼接成一个 NumPy 数组，并保存为 .csv 文件。

    参数:
        args: 包含配置参数的对象
        site_name (str): 站点名称
        timestamp_start (tuple): 起始时间戳
        timestamp_end (tuple): 结束时间戳

    返回:
        None
    """
    variable_mapping = {
        '2m_temperature': 't2m',
        '10m_u_component_of_wind': 'u10',
        '10m_v_component_of_wind': 'v10',
        'total_precipitation': 'tp',
        'surface_pressure': 'sp',
        'soil_temperature_level_1': 'stl1',
        'volumetric_soil_water_layer_1': 'swvl1',
        'surface_latent_heat_flux': 'slhf',
        'surface_net_solar_radiation': 'ssr',
        'surface_sensible_heat_flux': 'sshf',
        'total_evaporation': 'e',
        'leaf_area_index_high_vegetation': 'lai_hv',
        'leaf_area_index_low_vegetation': 'lai_lv',
        '2m_dewpoint_temperature': 'd2m',
        "surface_net_thermal_radiation": 'str',
        "surface_solar_radiation_downwards": 'ssrd',
        "surface_thermal_radiation_downwards": 'strd'

    }
    aux_data_folder = os.path.join(args.era5land_data_path, site_name)
    if not os.path.exists(aux_data_folder):
        raise FileNotFoundError(f"The folder '{aux_data_folder}' does not exist.")
    date_list = generate_date_combinations(timestamp_start, timestamp_end)
    if not date_list:
        raise ValueError("No date combinations generated. Please check the timestamp range.")
    # 用于存储所有变量的数据
    all_data = {}
    valid_times = []
    for var in args.aux_vars:
        if var not in variable_mapping:
            print(f'The variable {var} not found !!!')
            continue
        internal_var = variable_mapping[var]
        var_data = []
        for year, month in date_list:
            month = str(month).zfill(2)
            file_name = f'{site_name}_ERA5LAND_{var}_{year}_{month}.nc'
            file_path = os.path.join(aux_data_folder, file_name)
            try:
                # 使用 xarray 打开 .nc 文件
                ds = xr.open_dataset(file_path)
                print(f"Now we have loaded the file {file_name}")
                if internal_var not in ds.variables:
                    print(f"Variable '{internal_var}' not found in '{file_path}'. Skipping.")
                    ds.close()
                    continue
                # 提取变量数据
                data_array = ds[internal_var].values  # 转换为 NumPy 数组
                data_array = np.squeeze(data_array)
                var_data.append(data_array)
                # 如果是 surface_net_solar_radiation，提取 valid_time
                if var == 'surface_net_thermal_radiation':
                    valid_time = ds['valid_time'].values
                    valid_times.append(valid_time)
                ds.close()
            except Exception as e:
                print(f"Failed to read '{file_path}': {e}")
                continue
        # 拼接数据
        try:
            concatenated_data = np.concatenate(var_data)  # 在时间维度上进行拼接
            all_data[var] = concatenated_data
        except Exception as e:
            print(f'Error: {e}')
    # 拼接 valid_time
    if valid_times:
        concatenated_valid_times = np.concatenate(valid_times)
        # 将 Unix 时间戳转换为指定格式的字符串
        valid_time_formatted = pd.to_datetime(concatenated_valid_times, unit='s').strftime(args.timestamp_format)
        all_data['TIMESTAMP_START'] = valid_time_formatted
    # 保存为 .csv 文件
    df = pd.DataFrame(all_data)
    csv_file_name = f'{site_name}_Era5land_all.csv'
    outputfile_path = os.path.join(args.ground_data_path, source, site_name)
    os.makedirs(outputfile_path, exist_ok=True)
    output_file = os.path.join(outputfile_path, csv_file_name)
    try:
        df.to_csv(output_file, index=True)
        print(f"The auxiliary variable of {site_name} has been saved in {outputfile_path}, named {csv_file_name}")
    except Exception as e:
        print(f"Failed to save .csv file: {e}")
    return df


def set_stats ( args, label, aux ):
    var_orig = label.copy()
    label.index = pd.to_datetime(label.index)
    new_label = label.interpolate()
    original_start = new_label.index.min()
    original_end = new_label.index.max()
    original_freq = args.temporal_resolution
    new_index = pd.date_range(start=original_start, end=original_end, freq=original_freq)
    # -------------------------------------------------
    # set max
    flux_max = new_label.resample("D").max()
    flux_max_new = flux_max.reindex(new_index, method='bfill')
    aux["flux_max"] = flux_max_new[args.filling_var]  #
    # -------------------------------------------------
    # set min
    flux_min = new_label.resample("D").min()
    aux["flux_min"] = flux_min.reindex(new_index, method='bfill')
    aux["flux_min"] = aux["flux_min"]
    # -------------------------------------------------
    # set mean
    flux_mean = new_label.resample("D").mean()
    aux["flux_mean"] = flux_mean.reindex(new_index, method='bfill')
    aux["flux_mean"] = aux["flux_mean"]  # .ffill()
    # -------------------------------------------------
    # set std
    flux_std = new_label.resample("D").std()
    aux["flux_std"] = flux_std.reindex(new_index, method='bfill')
    aux["flux_std"] = aux["flux_std"]  # .ffill()
    # -------------------------------------------------
    # set 25%, 50%, 75% quantiles
    # ----------------------------
    # 25%:
    flux_p25 = new_label.resample("D").quantile(0.25)
    aux["flux_p25"] = flux_p25.reindex(new_index, method='bfill')
    aux["flux_p25"] = aux["flux_p25"]  # .ffill()
    # ----------------------------
    # 50%:
    flux_p50 = new_label.resample("D").quantile(0.50)
    aux["flux_p50"] = flux_p50.reindex(new_index, method='bfill')
    aux["flux_p50"] = aux["flux_p50"]  # .ffill()
    # ----------------------------
    # 75%:
    flux_p75 = new_label.resample("D").quantile(0.75)
    aux["flux_p75"] = flux_p75.reindex(new_index, method='bfill')
    aux["flux_p75"] = aux["flux_p75"]  # .ffill()
    aux = aux.interpolate()
    label = var_orig
    return label, aux, ["flux_max", "flux_min", "flux_mean", "flux_std", "flux_p25", "flux_p50", "flux_p75"]


def set_time_serise_tag (aux):
    aux = aux.copy()
    aux.loc[:, "year"] = aux.index.year
    aux.loc[:, "doy"] = aux.index.map(
        lambda x: int(x.strftime("%j"))
    )
    return aux


def add_time_series_feature ( args, train_data ):
    all_label = []
    all_aux = []
    all_mask = []
    for site_name, data in train_data.items():
        data['label'].set_index('TIMESTAMP_START', inplace=True)
        data['label'].index = pd.to_datetime(data['label'].index)
        data['aux_data'].index = pd.to_datetime(data['label'].index)
        label, aux, stat_tags = set_stats(args, data['label'], data['aux_data'])
        aux, doy_tag = set_time_serise_tag(label, aux)
        all_label.append(label[args.filling_var])
        all_aux.append(aux[args.aux_vars + stat_tags + doy_tag])
        all_mask.append(data['mask'])
    all_label = pd.concat(all_label, axis=0).to_numpy()
    all_aux = pd.concat(all_aux, axis=0).to_numpy()
    all_mask = np.concatenate(all_mask, axis=0)
    all_label = all_label[all_mask]
    all_aux = all_aux[all_mask]
    return all_label, all_aux


def normalize_data ( data, normalize_type='z-score', apply_mode='train', params=None ):
    """
    支持训练/测试模式的多维数据归一化函数
    参数说明：
    data: np.ndarray - 输入数据（支持任意维度）
    normalize_type: str - 归一化类型 ['z-score', 'min-max', 'max']
    feature_axis: tuple - 计算统计量的聚合轴（默认前两轴作为样本+时间轴）
    apply_mode: str - 工作模式 ['train' | 'test']
    params: dict - 测试模式时传入的训练集参数

    返回：
    normalized_data: np.ndarray - 归一化后数据
    params: dict - 训练模式返回参数，测试模式返回空字典
    """
    # 模式验证
    if apply_mode not in ['train', 'test']:
        raise ValueError("apply_mode must be 'train' or 'test'")

    # 测试模式必须传入参数
    if apply_mode == 'test' and params is None:
        raise ValueError("Test mode requires normalization parameters")

    # 自动维度检测
    original_shape = data.shape
    ndim = data.ndim

    # 重塑数据为2D格式 (samples, features)
    if ndim > 2:
        data_2d = data.reshape(-1, original_shape[-1])
    else:
        data_2d = data.reshape(len(data), -1)

    # 训练模式逻辑
    if apply_mode == 'train':
        # 根据类型选择归一化器
        if normalize_type == 'z-score':
            scaler = StandardScaler()
            scaler.fit(data_2d)
            normalized_2d = scaler.transform(data_2d)
            params = {
                'mean': scaler.mean_.reshape(1, -1),
                'scale': scaler.scale_.reshape(1, -1)
            }
        elif normalize_type == 'min-max':
            scaler = MinMaxScaler()
            scaler.fit(data_2d)
            normalized_2d = scaler.transform(data_2d)
            params = {
                'min': scaler.data_min_.reshape(1, -1),
                'range': scaler.data_range_.reshape(1, -1)
            }
        elif normalize_type == 'max':
            scaler = MaxAbsScaler()
            scaler.fit(data_2d)
            normalized_2d = scaler.transform(data_2d)
            params = {
                'max': scaler.max_abs_.reshape(1, -1)
            }
        else:
            raise ValueError(f"Unsupported normalization type: {normalize_type}")

    # 测试模式逻辑
    else:
        # 验证参数完整性
        required_params = {
            'z-score': ['mean', 'scale'],
            'min-max': ['min', 'range'],
            'max': ['max']
        }[normalize_type]

        if not all(k in params for k in required_params):
            missing = [k for k in required_params if k not in params]
            raise ValueError(f"Missing parameters: {missing}")

        # 应用参数转换
        if normalize_type == 'z-score':
            normalized_2d = (data_2d - params['mean']) / params['scale']
        elif normalize_type == 'min-max':
            normalized_2d = (data_2d - params['min']) / params['range']
        elif normalize_type == 'max':
            normalized_2d = data_2d / params['max']
        else:
            raise ValueError(f"Unsupported normalization type: {normalize_type}")
    # 恢复原始形状
    normalized_data = normalized_2d.reshape(original_shape)
    # 训练模式处理参数维度
    if apply_mode == 'train':
        # 自动广播参数到原始维度
        for key in params:
            param = params[key]
            if ndim > 2:
                param = param.reshape((1,) * (ndim - 2) + param.shape)
            params[key] = param
        return normalized_data, params
    # 测试模式不返回参数
    return normalized_data, {}


def denormalize ( data, params, normalize_type ):
    """反归一化核心逻辑"""
    original_shape = data.shape
    data_2d = data.reshape(-1, original_shape[-1])

    # 根据归一化类型应用逆运算
    if normalize_type == 'z-score':
        denorm_2d = data_2d * params['scale'] + params['mean']
    elif normalize_type == 'min-max':
        denorm_2d = data_2d * params['range'] + params['min']
    elif normalize_type == 'max':
        denorm_2d = data_2d * params['max']
    else:
        raise ValueError(f"Unsupported denormalization type: {normalize_type}")

    return denorm_2d.reshape(original_shape)
