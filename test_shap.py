from config.cfg import get_config
from models.model import GapfillModel, Transformer_NoEmbed, GapfillModel_NoEmbed, Transformer_GapfillModel
from utils.load_utils import load_site_information
from data.dataset import Gapfill_Val_Dataset,Gapfill_Train_Dataset
import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

os.environ["OMP_NUM_THREADS"] = '2'


class GapfillShapAnalyzer:
    def __init__ ( self, model, args ):
        self.model = model
        self.args = args
        self.model.eval()
        # 基础配置
        self.seq_len = args.seq_len
        self.n_cont = args.feature_in
        # 状态存储
        self.dims = {}
        self.idx_map = {}
        self.structure_type = None
        self.len_hist = 0
        self.len_curr = 0
        self.len_val = 0
        self.len_mask = 0
        self.len_qual = 0

    def detect_structure ( self, item ):

        if self.dims: return

        if isinstance(item, (tuple, list)) and len(item) > 0 and isinstance(item[0], dict):
            self.structure_type = 'dict_tuple'
            inputs = item[0]
            # 直接从字典里获取维度，不需要猜
            self.dims['hist_total'] = inputs['driver_history'].shape[-1]
            self.dims['curr_total'] = inputs['driver_current'].shape[-1]
            # 计算长度
            self.len_hist = int(np.prod(inputs['driver_history'].shape))
            self.len_curr = int(np.prod(inputs['driver_current'].shape))
            self.len_val = int(np.prod(inputs['input_values'].shape))
            self.len_qual = int(np.prod(inputs['input_quality'].shape))
            self.len_mask = 1  # mask 始终为 1
            print(f"  -> Dimensions: History={self.dims['hist_total']}, Current={self.dims['curr_total']}")

        elif isinstance(item, dict):
            self.structure_type = 'dict'
            # (同上)
            self.dims['hist_total'] = item['driver_history'].shape[-1]
            self.dims['curr_total'] = item['driver_current'].shape[-1]
            self.len_hist = int(np.prod(item['driver_history'].shape))
            self.len_curr = int(np.prod(item['driver_current'].shape))
            self.len_val = int(np.prod(item['input_values'].shape))
            self.len_qual = int(np.prod(item['input_quality'].shape))
            self.len_mask = 1
        else:

            self.structure_type = 'flat_tuple'
            data_list = list(item)
            # ... (保留原有的形状探测逻辑) ...
            for i, d in enumerate(data_list):
                if not torch.is_tensor(d): continue
                shape = d.shape
                if len(shape) == 2 and shape[0] == self.seq_len and shape[1] >= self.n_cont:
                    if shape[1] > 1 or self.n_cont == 1:
                        if 'history' not in self.idx_map or shape[1] > self.dims.get('hist_total', 0):
                            self.idx_map['history'] = i
                            self.dims['hist_total'] = shape[1]
                # Values
                elif len(shape) == 2 and shape[0] == self.seq_len and shape[1] == 1:
                    if 'values' not in self.idx_map: self.idx_map['values'] = i
                # Current
                elif len(shape) == 1 and shape[0] >= self.n_cont:
                    self.idx_map['current'] = i
                    self.dims['curr_total'] = shape[0]
                # Mask
                elif len(shape) == 1 and shape[0] == 1:
                    if 'mask' not in self.idx_map: self.idx_map['mask'] = i

            # 校验
            if 'history' not in self.idx_map: raise ValueError("Failed to detect History tensor")

            self.len_hist = self.seq_len * self.dims['hist_total']
            self.len_val = self.seq_len * 1
            self.len_curr = self.dims['curr_total']
            self.len_mask = 1


    def flatten_sample ( self, item ):
        """
        根据探测到的结构，提取并扁平化数据
        """
        if not self.dims: self.detect_structure(item)

        def to_np ( x ):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else x

        # === 分情况提取数据 ===
        if self.structure_type == 'dict_tuple':
            # item[0] 是 input 字典
            inputs = item[0]
            d_hist = to_np(inputs['driver_history']).flatten()
            d_val = to_np(inputs['input_values']).flatten()
            d_curr = to_np(inputs['driver_current']).flatten()
            d_input_quality = to_np(inputs['input_quality']).flatten()
            d_mask = to_np(inputs['night_mask']).flatten()
        elif self.structure_type == 'dict':
            d_hist = to_np(item['driver_history']).flatten()
            d_val = to_np(item['input_values']).flatten()
            d_curr = to_np(item['driver_current']).flatten()
            d_input_quality = to_np(item['input_quality']).flatten()
            d_mask = to_np(item['night_mask']).flatten()
        else:  # flat_tuple
            d_hist = to_np(item[self.idx_map['history']]).flatten()
            d_val = to_np(item[self.idx_map['values']]).flatten()
            d_curr = to_np(item[self.idx_map['current']]).flatten()
            d_mask = to_np(item[self.idx_map['mask']]).flatten() if 'mask' in self.idx_map else np.array([0.0])
            d_input_quality = to_np(item['input_quality']).flatten()

        return np.concatenate([d_hist, d_val, d_curr, d_input_quality, d_mask])

    def predict_wrapper ( self, x_numpy ):
        batch_size = x_numpy.shape[0]
        cursor = 0
        # 1. Slicing
        end = cursor + self.len_hist
        raw_hist = x_numpy[:, cursor:end]
        cursor = end
        end = cursor + self.len_val
        raw_val = x_numpy[:, cursor:end];
        cursor = end
        end = cursor + self.len_curr;
        raw_curr = x_numpy[:, cursor:end];
        cursor = end
        end = cursor + self.len_qual
        raw_qual = x_numpy[:, cursor: end]
        raw_night_mask = x_numpy[:, end:end + 1]
        # 2. Reassemble & Protect Discrete
        n_hist_total = self.dims['hist_total']
        t_hist_raw = torch.tensor(raw_hist, dtype=torch.float32).view(batch_size, self.seq_len, n_hist_total)
        t_hist = torch.cat([t_hist_raw[..., :self.n_cont], torch.round(t_hist_raw[..., self.n_cont:])], dim=-1)
        t_curr_raw = torch.tensor(raw_curr, dtype=torch.float32)
        t_curr = torch.cat([t_curr_raw[:, :self.n_cont], torch.round(t_curr_raw[:, self.n_cont:]).long().float()],
                           dim=1)
        t_val = torch.tensor(raw_val, dtype=torch.float32).view(batch_size, self.seq_len, 1)
        t_mask = torch.round(torch.tensor(raw_night_mask, dtype=torch.float32).view(batch_size, 1)).long()
        t_quality = torch.round(torch.tensor(raw_qual).view(batch_size, self.seq_len, 1)).long()
        input_data = {'driver_history': t_hist, 'driver_current': t_curr, 'input_values': t_val,
                      'input_quality': t_quality, 'night_mask': t_mask}
        with torch.no_grad():
            output = self.model(input_data)
        # 3. Shape Fix
        if output.ndim == 3: output = output[:, -1, :]
        if output.ndim == 1: output = output.unsqueeze(1)
        return output.detach().cpu().numpy()

    def plot_results ( self, shap_values, save_dir, sample_indices=None ):
        """
        聚合 SHAP 值，并为每个样本单独保存一份特征贡献记录
        """
        # 1. 维度修复 (确保是 [Batch, Total_Features])
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)
        # 2. 切片与聚合 (保持与 predict_wrapper 一致的顺序)
        cursor = 0
        # --- A. History (历史驱动) -> 按样本聚合时间步 ---
        shap_hist = shap_values[:, cursor: cursor + self.len_hist]
        shap_hist_3d = shap_hist.reshape(-1, self.seq_len, self.dims['hist_total'])
        shap_hist_sum = np.sum(shap_hist_3d, axis=1)  # [Batch, Hist_Total]
        cursor += self.len_hist
        # --- B. Values (历史目标值) ---
        shap_val = shap_values[:, cursor: cursor + self.len_val]
        shap_val_sum = np.sum(shap_val, axis=1, keepdims=True)  # [Batch, 1]
        cursor += self.len_val
        # --- C. Current (当前时刻) ---
        shap_curr = shap_values[:, cursor: cursor + self.len_curr]  # [Batch, Curr_Total]
        cursor += self.len_curr
        # --- D. Quality (数据质量标记) ---
        shap_qual = shap_values[:, cursor: cursor + self.len_qual]
        shap_qual_sum = np.sum(shap_qual, axis=1, keepdims=True)  # [Batch, 1]
        cursor += self.len_qual
        # --- E. Mask (夜间标记) ---
        shap_mask = shap_values[:, cursor: cursor + self.len_mask]  # [Batch, 1]
        # 这里的逻辑保持你原始代码的驱动变量合并：History + Current
        driver_shap = shap_hist_sum + shap_curr
        # 3. 拼接最终矩阵 [n_test, n_features_merged]
        final_matrix = np.concatenate([
            driver_shap,
            shap_val_sum,
            shap_qual_sum,
            shap_mask
        ], axis=1)
        # 4. 生成变量名
        cont_names = self.args.aux_variables
        names_merged_cont = [f"Total_{n}" for n in cont_names]
        names_merged_disc = ['Total_TOD', 'Total_DOY', 'Total_SiteID', 'Total_IGBP']
        feature_names = names_merged_cont + names_merged_disc + \
                        ["HistSum_Target", "HistSum_Quality", "Night_Mask"]
        # 维度对齐校验
        if len(feature_names) != final_matrix.shape[1]:
            feature_names = feature_names[:final_matrix.shape[1]]
            if len(feature_names) < final_matrix.shape[1]:
                feature_names += [f"Unknown_{i}" for i in range(final_matrix.shape[1] - len(feature_names))]
        # =========================================================
        # 5. 保存逐样本 CSV (每行是一个样本的详细贡献)
        # =========================================================
        print(f"[Info] Saving per-sample SHAP analysis...")
        # 创建 DataFrame
        df_individual = pd.DataFrame(final_matrix, columns=feature_names)
        # 如果提供了样本索引，将其插入到第一列方便溯源
        if sample_indices is not None:
            df_individual.insert(0, 'Original_Dataset_Index', sample_indices)
        csv_path = os.path.join(save_dir, f'shap_values_individual_{self.args.filling_var}.csv')
        df_individual.to_csv(csv_path, index=False)
        print(f"[Success] Individual SHAP values saved to: {csv_path}")
        print(f"Matrix shape: {final_matrix.shape} (Samples x Features)")


def load_model_weights ( model, args, target_mean=None, target_std=None ):
    """
    加载模型权重，并支持强制覆盖 RevIN 的统计值
    Args:
        model: 模型实例
        args: 配置参数
        target_mean: (可选) 预先计算好的全局均值
        target_std: (可选) 预先计算好的全局标准差
    """
    model_dir = os.path.join(args.output_path, args.filling_var,f"exp_{args.exp_num}")
    loss_suffix = "" if args.enable_QC_loss else "_withoutQCLoss"
    model_name = f'{args.model_name}_{args.filling_var}_best_model_{args.seq_len}_exp{args.exp_num}{loss_suffix}.pth'
    best_model_path = os.path.join(model_dir, model_name)

    if os.path.exists(best_model_path):
        print(f"[Info] Loading weights from: {best_model_path}")
        try:
            state_dict = torch.load(best_model_path, weights_only=True)
            if target_mean is not None and target_std is not None:
                new_mean = torch.as_tensor(target_mean, dtype=torch.float32).view(-1)
                new_std = torch.as_tensor(target_std, dtype=torch.float32).view(-1)
                state_dict['revin2.global_mean'] = new_mean
                state_dict['revin2.global_std'] = new_std
            else:
                keys_to_fix = ['revin2.global_mean', 'revin2.global_std']
                for key in keys_to_fix:
                    if key in state_dict and state_dict[key].ndim == 0:
                        state_dict[key] = state_dict[key].unsqueeze(0)
            # =====================================================
            model.load_state_dict(state_dict)
            model.eval()
            return True
        except Exception as e:
            print(f"[Error] Failed to load weights: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"[Warning] Weight file not found at {best_model_path}")
        return False


def main ( args ):
    print(f"\n=== Starting SHAP Analysis for {args.filling_var} ===")
    # 2. 加载站点元数据
    site_information = load_site_information(args.site_info_path)
    if site_information is None:
        raise ValueError("[Error] Failed to load site information.")
    site_names = site_information['SITE_ID'].unique().tolist()
    site_igbp = site_information['IGBP'].unique().tolist()
    unique_sites_num = len(site_names)
    unique_igbp_num = len(site_igbp)
    site_name_to_id = {name: i for i, name in enumerate(site_names)}
    igbp_to_id = {igbp_val: i for i, igbp_val in enumerate(site_igbp)}
    gap_fill_train_dataset = Gapfill_Train_Dataset(args, site_name_to_id, igbp_to_id, site_information)
    global_mean, global_std = gap_fill_train_dataset.calculate_global_stats()
    args.default_mean = global_mean
    args.default_std = global_std
    # 3. 初始化模型
    if args.model_name == "MOETransformer":
        print("[Info] Initializing MOETransformer...")
        model = GapfillModel(args, unique_sites_num, unique_igbp_num, 'test')
    elif args.model_name == 'Transformer':
        model = Transformer_GapfillModel(args, unique_sites_num, unique_igbp_num, "test")
    elif args.model_name == "MOETransformer_NoEmbed":
        model = GapfillModel_NoEmbed(args, "test")
    else:
        raise ValueError(f'[Error] Model {args.model_name} not defined!')
    # 4. 加载权重
    if not load_model_weights(model, args,args.default_mean,args.default_std):
        print("[Warning] Continuing with random weights (Analysis will be meaningless if not intended).")
    # 5. 准备数据
    print("[Info] Loading Validation Dataset...")
    gap_fill_val_dataset = Gapfill_Val_Dataset(args, scalers=None)
    print(f"[Info] Dataset loaded with {len(gap_fill_val_dataset)} samples.")
    print("\n=== Initializing SHAP Analyzer ===")
    analyzer = GapfillShapAnalyzer(model, args)
    # 参数设置
    n_bg = 500  # 背景样本数 (Baseline)
    n_test = 50  # 要解释的样本数 (Explain)
    n_samples = 1000 # 扰动次数
    print(f"[Info] Sampling {n_bg} background and {n_test} test samples...")
    try:
        _ = analyzer.flatten_sample(gap_fill_val_dataset[0])
    except Exception as e:
        print(f"[Error] Failed during structure detection: {e}")
        return
    # 2. 准备数据
    bg_indices = np.random.choice(len(gap_fill_val_dataset), min(n_bg, len(gap_fill_val_dataset)), replace=False)
    bg_data = np.array([analyzer.flatten_sample(gap_fill_val_dataset[i]) for i in bg_indices])
    # 使用kmeans
    kmeans_data = shap.kmeans(bg_data, 100).data
    test_indices = np.random.choice(len(gap_fill_val_dataset), min(n_test, len(gap_fill_val_dataset)), replace=False)
    test_data = np.array([analyzer.flatten_sample(gap_fill_val_dataset[i]) for i in test_indices])
    # 3. 计算 SHAP
    print("[Info] Running KernelExplainer (This may take a few minutes)...")
    explainer = shap.KernelExplainer(analyzer.predict_wrapper, kmeans_data)
    shap_values = explainer.shap_values(test_data, nsamples=n_samples)
    # 4. 保存结果
    save_dir = os.path.join(args.output_path, args.filling_var, "shap_analysis")
    os.makedirs(save_dir, exist_ok=True)
    analyzer.plot_results(shap_values, save_dir)
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    args = get_config()
    main(args)
