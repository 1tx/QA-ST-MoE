import calendar
import os
from datetime import date, timedelta
import pandas as pd
import xarray as xr
from multiprocessing import Pool
import psutil

# 新增：错误日志文件路径
ERROR_LOG_FILE = "hdf_error_log.txt"


def log_hdf_error ( error_msg, file_path ):
    """记录HDF错误到日志文件"""
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"Error: {error_msg}\nFile: {file_path}\n\n")


# this is my data process data
def generate_date_combinations ( start_year, start_month, end_year, end_month ):
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


def generate_site_data ( site_data, var, era5land_data_path, output_dir ):
    variable_name = {
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
    basename = os.path.basename(era5land_data_path)
    try:
        # 使用with语句确保文件正确关闭
        with xr.open_dataset(era5land_data_path) as era5land_data:
            for index, row in site_data.iterrows():
                site_id = row['SITE_ID']
                lat = row['LOCATION_LAT']
                lon = row['LOCATION_LONG']
                out_path = os.path.join(output_dir, site_id)
                os.makedirs(out_path, exist_ok=True)
                out_file_name = f'{site_id}_{basename}'
                out_file = os.path.join(out_path, out_file_name)
                # if os.path.exists(out_file):
                #     print(f'File already exists: {out_file}')
                #     continue
                era_site_data = era5land_data.sel(latitude=lat, longitude=lon, method='nearest')
                era_site_data['latitude'] = lat
                era_site_data['longitude'] = lon
                if variable_name[var] in era_site_data.data_vars:
                    era_site_data[variable_name[var]] = era_site_data[variable_name[var]].expand_dims(
                        dim=['latitude', 'longitude'])  # !!!!!!!!!!!!
                # 使用临时文件确保写入完整性
                temp_file = out_file + '.tmp'
                try:
                    era_site_data.to_netcdf(out_file, mode='w', format='NETCDF4')
                    # os.rename(temp_file, out_file)  # 原子操作
                    print(f'The {site_id} process over!!!')
                except Exception as e:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    raise e
                finally:
                    if 'era_site_data' in locals():
                        era_site_data.close()
    except Exception as e:
        error_msg = str(e)
        print(f'Error processing file {era5land_data_path}: {error_msg}')
        # 检查是否为HDF错误
        if "HDF error" in error_msg or "HDF5" in error_msg:
            log_hdf_error(error_msg, era5land_data_path)
    finally:
        # 获取当前进程的内存使用情况
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f'Process {os.getpid()} memory usage: {mem_info.rss / (1024 * 1024):.2f} MB')


def extract_data_form_era5land ( site_data, input_dir, output_dir, year ):
    """
    :param site_data:
    :param input_dir:
    :param output_dir:
    """
    print('Now we extract site data from Era5Land data')
    if year != 2024:
        date_combinations = generate_date_combinations(year, 9, year, 9)
    else:
        date_combinations = generate_date_combinations(year, 1, year, 9)
    tasks = []
    for year, month in date_combinations:
        for var in variable:
            input_file_path = input_dir
            if var == 'total_precipitation':
                year = str(year)
                month = str(month).zfill(2)
                input_file_name = f'Era5Land_{var}_{year}_{month}.nc'
                input_file = os.path.join(input_file_path, input_file_name)
                if not os.path.exists(input_file):
                    input_file_name = f'Era5Land_{var}_{year}_{month}.nc4'
                    input_file = os.path.join(input_file_path, input_file_name)
                    if not os.path.exists(input_file):
                        print(f'{input_file} not exists')
                        continue
            else:
                year = str(year)
                month = str(month).zfill(2)
                input_file_name = f'ERA5LAND_{var}_{year}_{month}.nc'
                input_file = os.path.join(input_file_path, input_file_name)
                if not os.path.exists(input_file):
                    input_file_name = f'ERA5LAND_{var}_{year}_{month}.nc4'
                    input_file = os.path.join(input_file_path, input_file_name)
                    if not os.path.exists(input_file):
                        print(f'{input_file} not exists')
                        continue
            tasks.append((site_data, var, input_file, output_dir))

    with Pool(processes=4) as pool:  # 创建一个包含 8 个进程的进程池
        pool.starmap(generate_site_data, tasks)  # 并行执行 tasks 中的每个任务


if __name__ == '__main__':
    # 初始化错误日志文件
    if os.path.exists(ERROR_LOG_FILE):
        os.remove(ERROR_LOG_FILE)

    site_data_path = '../data/config/test_select_site.xls'
    site_data = pd.read_excel(site_data_path, sheet_name='HR_site')  # HH_site,repeat_site,HR_site
    print(site_data.head())
    variable = [
        #  "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
       # "surface_net_thermal_radiation"
    ]
    for year in range(2021, 2022):
        input_dir = os.path.join('G:', 'Era5Land', str(year))
        output_dir = os.path.join('E:', 'ERA5LAND_FOR_SITE')
        os.makedirs(output_dir, exist_ok=True)
        extract_data_form_era5land(site_data, input_dir, output_dir, year)
