# -*- coding: utf-8 -*-
from debias_tool_V1 import find_fluxnet_files
import os
import pandas as pd

# 定义需要分析的变量及其对应的QC变量
variables_with_qc = {
    'G_F_MDS': 'G_F_MDS_QC',
    'H_F_MDS': 'H_F_MDS_QC',
    'LE_F_MDS': 'LE_F_MDS_QC',
    'LW_IN_F_MDS': 'LW_IN_F_MDS_QC',
    'SW_IN_F_MDS': 'SW_IN_F_MDS_QC',
    'TA_F_MDS': 'TA_F_MDS_QC',
    'VPD_F_MDS': 'VPD_F_MDS_QC',
    'NEE_VUT_REF': 'NEE_VUT_REF_QC'
}

# 定义没有QC的变量
variables_without_qc = ['P', 'PA', 'RH', 'NETRAD', 'WS', 'NIGHT']

site_base_path = 'E:/Flux_output/'
source_to_folder = {
    "FLUXNET": "FLX",
    "AmeriFLUX": "AMF",
    "ICOS": "ICOS"
}

# 读取站点信息
site_data = pd.read_excel('../data/config/select_site_UTC.xlsx', sheet_name='HR_site')#HH_site,repeat_site,HR_site

# 创建一个列表来存储每个站点的变量缺失统计结果
missing_statistics = []

for index, row in site_data.iterrows():
    site_id = row['SITE_ID']
    source = row['SOURCE']
    site_path = os.path.join(site_base_path, source_to_folder[source], site_id)

    # 找到站点数据文件
    site_file = find_fluxnet_files(site_path, source_to_folder[source])
    if site_file is None:
        print(f"No data file found for site {site_id} from source {source}")
        continue

    # 读取站点数据
    site_data = pd.read_csv(site_file)

    # 统计带有QC变量的缺失情况
    for var, qc_var in variables_with_qc.items():
        if var in site_data.columns and qc_var in site_data.columns:
            # QC变量为0的数据作为有效输入
            valid_data_count = (site_data[qc_var] == 0).sum()
            total_data_count = len(site_data[var])
            missing_count = total_data_count - valid_data_count
            missing_percentage = (missing_count / total_data_count) * 100 if total_data_count > 0 else 0
            missing_statistics.append({
                'SITE_ID': site_id,
                'SOURCE': source,
                'VARIABLE': var,
                'total_count': total_data_count,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage
            })
        else:
            missing_statistics.append({
                'SITE_ID': site_id,
                'SOURCE': source,
                'VARIABLE': var,
                'total_count': None,
                'missing_count': None,
                'missing_percentage': None,
                'note': f"The site {site_id} from source {source} does not provide data for {var}."
            })
            print(f"The site {site_id} from source {source} does not provide data for {var}.")

    # 统计没有QC变量的缺失情况
    for var in variables_without_qc:
        if var in site_data.columns:
            # 不等于-9999的值作为非缺失变量
            valid_data_count = (site_data[var] != -9999).sum()
            total_data_count = len(site_data[var])
            missing_count = total_data_count - valid_data_count
            missing_percentage = (missing_count / total_data_count) * 100 if total_data_count > 0 else 0
            missing_statistics.append({
                'SITE_ID': site_id,
                'SOURCE': source,
                'VARIABLE': var,
                'total_count': total_data_count,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage
            })
        else:
            missing_statistics.append({
                'SITE_ID': site_id,
                'SOURCE': source,
                'VARIABLE': var,
                'total_count': None,
                'missing_count': None,
                'missing_percentage': None,
                'note': f"The site {site_id} from source {source} does not provide data for {var}."
            })
            print(f"The site {site_id} from source {source} does not provide data for {var}.")
    print(f'{site_id} from source {source} 统计完成')

# 将统计结果转换为DataFrame
missing_df = pd.DataFrame(missing_statistics)

# 保存到Excel文件
missing_df.to_excel('missing_statistics_HR_site.xlsx', index=False)
print("Missing data statistics have been saved to 'missing_statistics_HR_site.xlsx'.")