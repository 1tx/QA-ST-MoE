import pandas as pd
import os
from debias_tool import debias_era5land_HH, debias_era5land_HR


# 对站点数据进行去偏处理
def load_site_information ( sheet_name: str ) -> pd.DataFrame:
    # site information path
    site_information = '../data/config/select_site_UTC.xlsx'
    return pd.read_excel(site_information, sheet_name=sheet_name)


if __name__ == '__main__':
    source_to_folder = {
        "FLUXNET": "FLX",
        "AmeriFLUX": "AMF",
        "ICOS": "ICOS"
    }

    site_information = load_site_information('site_info')
    site_path_base = os.path.join('E:/', 'data')
    era_path_base = os.path.join('E:/', 'data_new')
    debias_file_path_base = os.path.join('E:/', 'data_new')
    # 用于全站汇总
    all_stats = []  # 去偏状态/参数
    all_eval = []  # 评估结果
    for index, row in site_information.iterrows():
        site_name = row['SITE_ID']
        site_source = row['SOURCE']
        site_time_start = row['START_TIME']
        site_time_end = row['END_TIME']
        site_resolution = row['TIME_RESOLUTION']
        site_path = os.path.join(site_path_base, source_to_folder[site_source], site_name)
        era_path = os.path.join(era_path_base, source_to_folder[site_source], site_name)
        debias_file_path = os.path.join(debias_file_path_base, source_to_folder[site_source], site_name)
        os.makedirs(debias_file_path, exist_ok=True)
        if site_resolution == 'HH':
            # 要求 debias_era5land_HH 返回: (debiased_df, stats_df, eval_df)
            stats_df, eval_df = debias_era5land_HH(
                site_path,era_path, site_name, source_to_folder[site_source],
                site_time_start, site_time_end,debias_file_path
            )
        elif site_resolution == 'HR':
            # 要求 debias_era5land_HR 也返回: (debiased_df, stats_df, eval_df)
            stats_df, eval_df = debias_era5land_HR(
                site_path,era_path, site_name, source_to_folder[site_source],
                site_time_start, site_time_end,debias_file_path
            )
            debiased_file_name = f'{site_name}_Era5land_HR_debias.csv'
        else:
            print(f"[WARN] {site_name}: unknown resolution {site_resolution}, skipped.")
            continue

        # ---- 规范化并追加站点名 ----
        # stats_df: index 是 era5_var，转成列；加 site_id
        if isinstance(stats_df.index, pd.Index):
            stats_df = stats_df.reset_index().rename(columns={'era5_var': 'variable'})
        if 'site_id' not in stats_df.columns:
            stats_df.insert(0, 'site_id', site_name)
        else:
            stats_df['site_id'] = site_name
        # eval_df: 确保有 site_id 列（有的实现已添加，这里统一覆盖为当前站点名）
        if 'site_id' not in eval_df.columns:
            eval_df.insert(0, 'site_id', site_name)
        else:
            eval_df['site_id'] = site_name

        all_stats.append(stats_df)
        all_eval.append(eval_df)
    # ---- 汇总保存为两个 CSV ----
    out_dir = os.path.join(debias_file_path_base, 'GLOBAL_OUTPUTS')
    os.makedirs(out_dir, exist_ok=True)
    if len(all_stats) > 0:
        all_stats_df = pd.concat(all_stats, ignore_index=True)
        all_stats_path = os.path.join(out_dir, 'ALL_SITES_Debias_Status.csv')
        all_stats_df.to_csv(all_stats_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Saved global debias status: {all_stats_path}")
    else:
        print("[INFO] No debias status to save.")

    if len(all_eval) > 0:
        all_eval_df = pd.concat(all_eval, ignore_index=True)
        all_eval_path = os.path.join(out_dir, 'ALL_SITES_Debias_Evaluation.csv')
        all_eval_df.to_csv(all_eval_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Saved global evaluation: {all_eval_path}")
    else:
        print("[INFO] No evaluation to save.")
