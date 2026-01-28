from utils.data_utils import generate_data, normalize_data, generate_test_data, denormalize_data
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Gapfill_Train_Dataset(Dataset):
    """
    Custom dataset class for loading and processing Gapfill data.
    """

    def __init__ ( self, args, site_name_to_id, igbp_to_id, select_site ):
        """
        Initialization function.
        :param args: Object containing configuration parameters.
        :param site_name_toid : a dict with site names as keys and their corresponding IDs as values.
        :param igbp_to_id: a dict with IGBP codes as keys and their corresponding IDs as values.
        :param select_site: DataFrame containing site information.
        """
        output_base = os.path.join(args.output_path, args.filling_var)
        observation_file = os.path.join(output_base, f"train_all_observation_{args.filling_var}.npy")
        driver_history_file = os.path.join(output_base, f"train_all_driver_history_{args.seq_len}.npy")
        driver_current_file = os.path.join(output_base, f'train_all_driver_current.npy')
        night_mask_file = os.path.join(output_base, f"train_all_night_mask_{args.filling_var}.npy")
        input_value = os.path.join(output_base, f"train_all_input_value_{args.filling_var}_{args.seq_len}.npy")
        quality = os.path.join(output_base, f"train_all_quality_{args.filling_var}_{args.seq_len}.npy")
        valid_mask = os.path.join(output_base, f"train_all_valid_mask.npy")
        path_to_check = [observation_file, driver_history_file, driver_current_file, night_mask_file, input_value,
                         quality, valid_mask]
        if not all(os.path.exists(p) for p in path_to_check):
            print('generate train data')
            observation, input_value, driver_history, driver_current, night_mask, quality, valid_mask = generate_data(
                args, site_name_to_id, igbp_to_id, select_site)
        else:
            # Load directly
            print(f'load train data')
            observation = np.load(observation_file)
            driver_history = np.load(driver_history_file)
            driver_current = np.load(driver_current_file)
            night_mask = np.load(night_mask_file)
            input_value = np.load(input_value)
            quality = np.load(quality)
            valid_mask = np.load(valid_mask)
        # Convert to tensor
        self.args = args
        self.driver_history = torch.tensor(driver_history, dtype=torch.float32)
        self.driver_current = torch.tensor(driver_current, dtype=torch.float32)
        self.observation = torch.tensor(observation, dtype=torch.float32)
        self.night_mask = torch.tensor(night_mask, dtype=torch.long)
        self.input_value = torch.tensor(input_value, dtype=torch.float32)
        self.quality = torch.tensor(quality, dtype=torch.long)
        self.valid_mask = torch.tensor(valid_mask, dtype=torch.long)
        # Normalization processing
        self.normalize = args.global_normalize
        self.scalers = {}
        if self.normalize:
            print("Applying Global Z-score Normalization on Train Data...")
            self._compute_and_apply_normalization()

    def _compute_and_apply_normalization ( self ):
        """
        计算统计量并对数据进行原地标准化 (In-place Normalization)
        """
        eps = 1e-6  # 防止除以0
        # 1. 计算 Observation 的统计量 (仅使用 valid mask = 0 的有效数据) input_value 和 observation 是同一物理变量，共用这组统计量
        valid_indices = (self.valid_mask == 0).view(-1)  # 展平
        if valid_indices.sum() > 0:
            obs_valid = self.observation.view(-1)[valid_indices]
            obs_mean = obs_valid.mean()
            obs_std = obs_valid.std()
        else:
            obs_mean = torch.tensor(0.0)
            obs_std = torch.tensor(1.0)

        self.scalers['obs'] = {'mean': obs_mean, 'std': obs_std}
        # 2. 计算 Drivers 的统计量
        num_cont = len(self.args.aux_variables)
        hist_cont_part = self.driver_history[:, :, :num_cont]
        driver_hist_mean = hist_cont_part.mean(dim=(0, 1))
        driver_hist_std = hist_cont_part.std(dim=(0, 1))
        # Driver Current 形状通常为 [N, Feat]
        curr_cont_part = self.driver_current[:, :num_cont]
        driver_curr_mean = curr_cont_part.mean(dim=0)  # [num_cont]
        driver_curr_std = curr_cont_part.std(dim=0)  # [num_cont]
        self.scalers['driver_hist'] = {'mean': driver_hist_mean, 'std': driver_hist_std}
        self.scalers['driver_curr'] = {'mean': driver_curr_mean, 'std': driver_curr_std}
        # 3. 应用标准化 (Z-score: (x - mean) / std)
        # Observation & Input Value
        self.observation = (self.observation - obs_mean) / (obs_std + eps)
        self.input_value = (self.input_value - obs_mean) / (obs_std + eps)
        # History
        self.driver_history[:, :, :num_cont] = (self.driver_history[:, :, :num_cont] - driver_hist_mean) / (
                driver_hist_std + eps)
        # Current
        self.driver_current[:, :num_cont] = (self.driver_current[:, :num_cont] - driver_curr_mean) / (
                driver_curr_std + eps)
        print(f"Normalization Done")

    def denormalize_obs ( self, tensor ):
        """反标准化 Observation (用于可视化或指标计算)"""
        if not self.normalize: return tensor
        mean = self.scalers['obs']['mean'].to(tensor.device)
        std = self.scalers['obs']['std'].to(tensor.device)
        return tensor * std + mean

    def __len__ ( self ):
        """
        Return the number of samples in the dataset.
        """
        return len(self.observation)

    def __getitem__ ( self, idx ):
        model_inputs = {
            "driver_history": self.driver_history[idx],
            "driver_current": self.driver_current[idx],
            "input_values": self.input_value[idx],
            "input_quality": self.quality[idx],
            "night_mask": self.night_mask[idx]
        }
        target = self.observation[idx]
        valid_mask = self.valid_mask[idx]
        return model_inputs, target, valid_mask

    def calculate_global_stats ( self ):
        """
        计算整个训练集中 Observation 的全局均值和标准差。
        关键点：只计算 valid_mask 为 0(有效) 的数据点。

        Returns:
            global_mean (float): 全局有效数据的均值
            global_std (float): 全局有效数据的标准差
        """
        print("Calculating global stats based on valid mask...")

        # 1. 确保 mask 是布尔类型，用于索引
        # valid_mask 形状通常是 [N, 1] 或 [N]

        mask_bool = self.valid_mask == 0
        # 确保 observation 和 valid_mask 维度匹配，防止广播错误
        if self.observation.shape != self.valid_mask.shape:
            # 尝试调整 mask 形状以匹配 observation
            mask_bool = mask_bool.view_as(self.observation)
        # 2. 使用布尔索引提取有效数据
        # 这会将数据展平为 1D Tensor，只包含有效值
        valid_data = self.observation[mask_bool]
        # 3. 边界情况处理：如果没有有效数据
        if valid_data.numel() == 0:
            print("[Warning] No valid data found in training set! Returning default 0 mean and 1 std.")
            return 0.0, 1.0
        # 4. 计算均值和标准差
        global_mean = torch.mean(valid_data).item()
        global_std = torch.std(valid_data).item()

        print(
            f"Global Stats Calculated -> Mean: {global_mean:.4f}, Std: {global_std:.4f}, Valid Count: {valid_data.numel()}")

        return global_mean, global_std


class Gapfill_Val_Dataset(Dataset):
    def __init__ ( self, args, scalers ):
        data_path = os.path.join(args.output_path, args.filling_var)
        observation_file = os.path.join(data_path, f"val_all_observation_{args.filling_var}.npy")
        driver_history_file = os.path.join(data_path, f"val_all_driver_history_{args.seq_len}.npy")
        driver_current_file = os.path.join(data_path, f"val_all_driver_current.npy")
        night_mask_file = os.path.join(data_path, f"val_all_night_mask_{args.filling_var}.npy")
        input_value_file = os.path.join(data_path, f"val_all_input_value_{args.filling_var}_{args.seq_len}.npy")
        quality = os.path.join(data_path, f"val_all_quality_{args.filling_var}_{args.seq_len}.npy")
        valid_mask = os.path.join(data_path, f'val_all_valid_mask.npy')
        path_to_check = [observation_file, driver_history_file, driver_current_file, night_mask_file, input_value_file,
                         quality, valid_mask]
        if not all(os.path.exists(p) for p in path_to_check):
            raise ValueError("The val file does not exist")
        else:
            # Load directly
            val_observation = np.load(observation_file)
            val_driver_history = np.load(driver_history_file)
            val_driver_current = np.load(driver_current_file)
            val_night_mask = np.load(night_mask_file)
            val_input_value = np.load(input_value_file)
            val_quality = np.load(quality)
            val_valid_mask = np.load(valid_mask)
        self.args = args
        self.driver_history = torch.tensor(val_driver_history, dtype=torch.float32)
        self.driver_current = torch.tensor(val_driver_current, dtype=torch.float32)
        self.observation = torch.tensor(val_observation, dtype=torch.float32)
        self.night_mask = torch.tensor(val_night_mask, dtype=torch.long)
        self.input_value = torch.tensor(val_input_value, dtype=torch.float32)
        self.quality = torch.tensor(val_quality, dtype=torch.long)
        self.valid_mask = torch.tensor(val_valid_mask, dtype=torch.long)
        self.normalize = args.global_normalize
        self.scalers = scalers
        if self.normalize and self.scalers is not None:
            self._apply_normalization()

    def _apply_normalization ( self ):
        """
        仅对连续变量部分 (前 num_cont 个特征) 应用标准化，
        保护后续的离散特征 (One-hot/Label) 不被修改。
        """
        eps = 1e-6
        # 1. 确定连续变量的特征数量 (必须与训练集一致)
        num_cont = len(self.args.aux_variables)
        # 2. 获取训练集的统计量
        obs_mean = self.scalers['obs']['mean']
        obs_std = self.scalers['obs']['std']
        # 注意：这里的 driver 统计量维度应该是 [num_cont]，因为训练集计算时已经切片了
        d_hist_mean = self.scalers['driver_hist']['mean']
        d_hist_std = self.scalers['driver_hist']['std']
        d_curr_mean = self.scalers['driver_curr']['mean']
        d_curr_std = self.scalers['driver_curr']['std']
        # 3. 应用标准化 - Observation & Input Value (针对单一变量，全量应用)
        # input_value 与 observation 是同一物理量
        self.observation = (self.observation - obs_mean) / (obs_std + eps)
        self.input_value = (self.input_value - obs_mean) / (obs_std + eps)
        # 4. 应用标准化 - Drivers (针对多变量，切片应用)
        # -----------------------------------------------------------
        # Driver History: [Batch, Seq, Feat]
        # mean/std 会自动广播匹配 Feat 维度
        self.driver_history[:, :, :num_cont] = (self.driver_history[:, :, :num_cont] - d_hist_mean) / (d_hist_std + eps)
        # Driver Current: [Batch, Feat]
        self.driver_current[:, :num_cont] = (self.driver_current[:, :num_cont] - d_curr_mean) / (d_curr_std + eps)
        print(f"Validation Data Normalized (First {num_cont} features).")

    def denormalize_obs ( self, tensor ):
        """反标准化辅助函数"""
        if not self.normalize or self.scalers is None: return tensor
        mean = self.scalers['obs']['mean'].to(tensor.device)
        std = self.scalers['obs']['std'].to(tensor.device)
        return tensor * std + mean

    def __len__ ( self ):
        """
        Return the number of samples in the dataset.
        """
        return len(self.observation)

    def __getitem__ ( self, idx ):
        model_inputs = {
            "driver_history": self.driver_history[idx],
            "driver_current": self.driver_current[idx],
            "input_values": self.input_value[idx],
            "input_quality": self.quality[idx],
            "night_mask": self.night_mask[idx]
        }
        target = self.observation[idx]
        valid_mask = self.valid_mask[idx]
        return model_inputs, target, valid_mask


class test_Gapfill_Dataset(Dataset):
    def __init__ ( self, args, site, site_igbp, site_name_to_id, scalers=None ):
        source_to_folder = {
            "FLUXNET": "FLX",
            "AmeriFLUX": "AMF",
            "ICOS": "ICOS"
        }
        site_name = site['SITE_ID']
        source = site['SOURCE']
        observation_path = os.path.join(args.output_path, args.filling_var, source_to_folder[source], site_name)
        observation_file = os.path.join(observation_path, f'{site_name}_{args.filling_var}_obs.npy')

        driver_history_file = os.path.join(args.output_path, 'driver', source_to_folder[source], site_name,
                                           f'{site_name}_all_driver_history_{args.seq_len}.npy')
        driver_current_file = os.path.join(args.output_path, 'driver', source_to_folder[source], site_name,
                                           f'{site_name}_all_driver_current.npy')
        night_mask_file = os.path.join(args.output_path, "night", source_to_folder[source], site_name,
                                       f'{site_name}_night_mask.npy')
        input_value_file = os.path.join(observation_path,
                                        f'{site_name}_{args.filling_var}_input_value_{args.seq_len}.npy')
        quality_file = os.path.join(observation_path, f"{site_name}_{args.filling_var}_quality_{args.seq_len}.npy")
        valid_mask_file = os.path.join(observation_path, f"{site_name}_{args.filling_var}_valid_mask.npy")
        path_to_check = [observation_file, driver_history_file, driver_current_file, night_mask_file, input_value_file,
                         quality_file,
                         valid_mask_file]
        if not all(os.path.exists(p) for p in path_to_check):
            observation, input_value, driver_history, driver_current, night_mask, quality, valid_mask = generate_test_data(
                args, site,
                site_igbp,
                site_name_to_id)
        else:
            observation = np.load(observation_file)
            driver_history = np.load(driver_history_file)
            driver_current = np.load(driver_current_file)
            night_mask = np.load(night_mask_file)
            input_value = np.load(input_value_file)
            quality = np.load(quality_file)
        self.driver_history = torch.tensor(driver_history, dtype=torch.float32)
        self.driver_current = torch.tensor(driver_current, dtype=torch.float32)
        self.observation = torch.tensor(observation, dtype=torch.float32)
        self.night_mask = torch.tensor(night_mask, dtype=torch.long)
        self.input_value = torch.tensor(input_value, dtype=torch.float32)
        self.quality = torch.tensor(quality, dtype=torch.long)
        self.normalize = args.global_normalize
        self.scalers = scalers
        self.args = args
        # 如果开启标准化且传入了scalers，则执行标准化
        if self.normalize and self.scalers is not None:
            self._apply_normalization()
        elif self.normalize and self.scalers is None:
            print("[Warning] Test dataset wants normalization but no 'scalers' provided! Data remains un-normalized.")

    def _apply_normalization ( self ):
        """
        仅对连续变量部分 (前 num_cont 个特征) 应用标准化，
        保护后续的离散特征 (One-hot/Label) 不被修改。
        """
        eps = 1e-6
        # 1. 确定连续变量的特征数量 (必须与训练集一致)
        num_cont = len(self.args.aux_variables)
        # 2. 获取训练集的统计量
        obs_mean = self.scalers['obs']['mean']
        obs_std = self.scalers['obs']['std']
        # 注意：这里的 driver 统计量维度应该是 [num_cont]，因为训练集计算时已经切片了
        d_hist_mean = self.scalers['driver_hist']['mean']
        d_hist_std = self.scalers['driver_hist']['std']
        d_curr_mean = self.scalers['driver_curr']['mean']
        d_curr_std = self.scalers['driver_curr']['std']
        # 3. 应用标准化 - Observation & Input Value (针对单一变量，全量应用)
        # input_value 与 observation 是同一物理量
        self.observation = (self.observation - obs_mean) / (obs_std + eps)
        self.input_value = (self.input_value - obs_mean) / (obs_std + eps)
        # 4. 应用标准化 - Drivers (针对多变量，切片应用)
        # -----------------------------------------------------------
        # Driver History: [Batch, Seq, Feat]
        # mean/std 会自动广播匹配 Feat 维度
        self.driver_history[:, :, :num_cont] = (self.driver_history[:, :, :num_cont] - d_hist_mean) / (d_hist_std + eps)
        # Driver Current: [Batch, Feat]
        self.driver_current[:, :num_cont] = (self.driver_current[:, :num_cont] - d_curr_mean) / (d_curr_std + eps)
        print(f"Validation Data Normalized (First {num_cont} features).")

    def denormalize ( self, tensor, var_type='obs' ):
        """
        反标准化函数：用于将模型输出变回真实物理值。

        Args:
            tensor (torch.Tensor): 标准化后的数据（例如模型的预测输出）
            var_type (str): 变量类型，默认为 'obs' (观测值/目标值)

        Returns:
            torch.Tensor: 原始物理单位的数据
        """
        if not self.normalize or self.scalers is None:
            return tensor

        if var_type == 'obs':
            mean = self.scalers['obs']['mean'].to(tensor.device)
            std = self.scalers['obs']['std'].to(tensor.device)
            return tensor * std + mean
        return tensor

    def __getitem__ ( self, idx ):
        model_inputs = {
            "driver_history": self.driver_history[idx],
            "driver_current": self.driver_current[idx],
            "input_values": self.input_value[idx],
            "input_quality": self.quality[idx],
            "night_mask": self.night_mask[idx]
        }
        target = self.observation[idx]
        return model_inputs, target

    def __len__ ( self ):
        return len(self.observation)
