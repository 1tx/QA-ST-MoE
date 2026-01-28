import os.path
import glob
import pandas as pd
from debias_tool_V1 import kelvin_to_celsius, calculate_vpd_and_RH, calculate_wind_speed, Pa_to_kPa, \
    convert_era5_accumulated_values

# 对原始Era5land数据进行单位转换，和时间戳对齐



source_to_folder = {
    "FLUXNET": "FLX",
    "AmeriFLUX": "AMF",
    "ICOS": "ICOS"
}


def find_era5_data ( data_path ):
    """
    在指定路径下查找符合命名格式 *_ERA5_all_HH.csv 的文件。
    确保只返回一个文件。如果找到多个文件或没有找到文件，则抛出异常。

    参数:
        data_path (str): 数据所在的文件夹路径。

    返回:
        str: 唯一符合条件的文件路径。

    异常:
        FileNotFoundError: 如果没有找到符合条件的文件。
        ValueError: 如果找到多个符合条件的文件。
    """
    # 构造文件匹配模式
    pattern = os.path.join(data_path, '*_Era5land_all.csv')

    # 使用 glob 模块查找符合条件的文件
    matching_files = glob.glob(pattern)

    # 检查找到的文件数量
    if len(matching_files) == 0:
        # 如果没有找到文件，抛出 FileNotFoundError 异常
        raise FileNotFoundError(f"No ERA5 data found in {data_path}")
    elif len(matching_files) > 1:
        # 如果找到多个文件，抛出 ValueError 异常
        raise ValueError(f"Multiple ERA5 data files found in {data_path}: {matching_files}")
    else:
        # 如果找到一个文件，返回文件路径
        return matching_files[0]


if __name__ == "__main__":
    output_path = "E:/data_new"
    site_data_path = '../data/config/select_site_UTC.xlsx'
    site_data = pd.read_excel(site_data_path, sheet_name='site_info')
    # 找到Era5land对应的数据
    for idx, row in site_data.iterrows():
        source = row["SOURCE"]
        site_id = row["SITE_ID"]
        site_UTC_offet = row['UTC_OFFSET']
        file_path = os.path.join(output_path, source_to_folder[source], site_id)
        os.makedirs(file_path,exist_ok=True)
        file_name = f'{site_id}_Era5land.csv'
        # find the Era5land data
        era5_data_path_root = os.path.join('E:/data', source_to_folder[source], site_id)
        era5_data_path = find_era5_data(era5_data_path_root)
        era5_data = pd.read_csv(era5_data_path)
        processed_data = pd.DataFrame()
        # convert temperature unit from k to c
        if '2m_temperature' and '2m_dewpoint_temperature' and 'soil_temperature_level_1' in era5_data.columns:
            processed_data['2m_temperature_C'] = kelvin_to_celsius(era5_data['2m_temperature'])
            processed_data['soil_temperature_level_1_C'] = kelvin_to_celsius(era5_data["soil_temperature_level_1"])
            processed_data['2m_dewpoint_temperature_C'] = kelvin_to_celsius(era5_data['2m_dewpoint_temperature'])
            processed_data['VPD'], processed_data['RH'] = calculate_vpd_and_RH(processed_data['2m_temperature_C'],
                                                                               processed_data[
                                                                                   '2m_dewpoint_temperature_C'])
        else:
            raise ValueError('temperature variable not exist ')
        print('Temperature unit already change from k to c')
        print(f'Vpd and RH of {site_id} have calculated ')
        # convert pressure unit from Pa to kPa
        if 'surface_pressure' in era5_data.columns:
            processed_data['surface_pressure'] = Pa_to_kPa(era5_data['surface_pressure'])
            print(f'surface_pressure unit of {site_id} already change from Pa to kPa')

        if '10m_u_component_of_wind' and '10m_v_component_of_wind' in era5_data.columns:
            processed_data['wind speed'] = calculate_wind_speed(era5_data['10m_u_component_of_wind'],
                                                                era5_data['10m_v_component_of_wind'])
            processed_data['wind 10m_u_component_of_wind'] = era5_data['10m_u_component_of_wind']
            processed_data['wind 10m_v_component_of_wind'] = era5_data['10m_v_component_of_wind']
            print(f'Wind speed of {site_id} have calculated!!')
        # convert_era5_accumulated_values
        if 'TIMESTAMP_START' and 'total_precipitation' and 'total_evaporation' in era5_data.columns:
            processed_data['total_precipitation'] = (
                convert_era5_accumulated_values(df=era5_data[['TIMESTAMP_START', 'total_precipitation']]
                                                , var_type='precipitation'))
        if 'TIMESTAMP_START' and 'total_evaporation' in era5_data.columns:
            processed_data['total_evaporation'] = (
                convert_era5_accumulated_values(df=era5_data[['TIMESTAMP_START', 'total_evaporation']]
                                                , var_type='flux'))

        if ('TIMESTAMP_START' and 'surface_latent_heat_flux' and 'surface_net_solar_radiation' and
                'surface_sensible_heat_flux' and 'surface_net_thermal_radiation' and 'surface_solar_radiation_downwards'
                and 'surface_thermal_radiation_downwards' in era5_data.columns):
            processed_data['surface_latent_heat_flux'] = convert_era5_accumulated_values(
                df=era5_data[['TIMESTAMP_START', 'surface_latent_heat_flux']]
                , var_type='radiation')
            processed_data['surface_sensible_heat_flux'] = convert_era5_accumulated_values(
                df=era5_data[['TIMESTAMP_START', 'surface_sensible_heat_flux']]
                , var_type='radiation')
            ssr = convert_era5_accumulated_values(
                df=era5_data[['TIMESTAMP_START', 'surface_net_solar_radiation']]
                , var_type='radiation')
            str_e = convert_era5_accumulated_values(
                df=era5_data[['TIMESTAMP_START', 'surface_net_thermal_radiation']]
                , var_type='radiation')
            processed_data['surface_net_solar_radiation'] = ssr
            processed_data['surface_net_thermal_radiation'] = str_e
            processed_data['surface_net_radiation'] = ssr + str_e
            print('The surface_net_radiation have calculated!! ')
            processed_data['surface_solar_radiation_downwards'] = convert_era5_accumulated_values(
                df=era5_data[['TIMESTAMP_START', 'surface_solar_radiation_downwards']]
                , var_type='radiation')
            processed_data['surface_thermal_radiation_downwards'] = convert_era5_accumulated_values(
                df=era5_data[['TIMESTAMP_START', 'surface_thermal_radiation_downwards']]
                , var_type='radiation')
        if 'leaf_area_index_high_vegetation' and 'leaf_area_index_low_vegetation' in era5_data.columns:
            processed_data['leaf_area_index_high_vegetation'] = era5_data['leaf_area_index_high_vegetation']
            processed_data['leaf_area_index_low_vegetation'] = era5_data['leaf_area_index_low_vegetation']
        if 'volumetric_soil_water_layer_1' in era5_data.columns:
            processed_data['volumetric_soil_water_layer_1'] = era5_data['volumetric_soil_water_layer_1']
        # 对于瞬时变量风速温度等TIMESTAMP_START为冗余时间戳并无实际含义，对于平均变量如辐射降雨等，TIMESTAMP_START表示开始时间,TIMESTAMP_END表示结束时间
        processed_data['TIMESTAMP_END'] = pd.to_datetime(era5_data['TIMESTAMP_START'],
                                                         format="%Y%m%d%H%M") + pd.Timedelta(hours=site_UTC_offet)
        processed_data['TIMESTAMP_START'] = processed_data['TIMESTAMP_END'] - pd.Timedelta(hours=1)
        # 由于累积变量的计算原因，需要将第一行数据剔除。
        processed_data = processed_data.drop(index=processed_data.index[0])
        processed_data.to_csv(os.path.join(file_path, file_name), index=False)
        print(f'{file_name} have saved to {file_path}')
