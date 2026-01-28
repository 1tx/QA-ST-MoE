import os
import re
import glob
import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# =============================
# Variable & rules
# =============================
FLUX_TO_ERA5LAND = {
    'TA_F_MDS': '2m_temperature_C',
    'VPD_F_MDS': 'VPD',
    'RH': 'RH',
    'WS': 'wind speed',
    'PA': 'surface_pressure',
    'P': 'total_precipitation',
    'LE_F_MDS': 'surface_latent_heat_flux',
    'H_F_MDS': 'surface_sensible_heat_flux',
    'NETRAD': 'surface_net_radiation',
    'LW_IN_F_MDS': 'surface_thermal_radiation_downwards',
    'SW_IN_F_MDS': 'surface_solar_radiation_downwards'
}

# Aggregation rules when resampling
RESAMPLE_RULE = {
    '2m_temperature_C': 'mean',
    'VPD': 'mean',
    'RH': 'mean',
    'wind speed': 'mean',
    'surface_pressure': 'mean',
    'total_precipitation': 'sum',
    'surface_latent_heat_flux': 'mean',
    'surface_sensible_heat_flux': 'mean',
    'surface_net_radiation': 'mean',
    'surface_thermal_radiation_downwards': 'mean',
    'surface_solar_radiation_downwards': 'mean',
}

# Simple physical bounds used for clipping debiased outputs
PHYSICAL_BOUNDS = {
    'RH': (0.0, 100.0),
    'VPD': (0.0, None),
    'total_precipitation': (0.0, None),
    'surface_solar_radiation_downwards': (0.0, None),
    'surface_net_radiation': (None, None),  # can be negative
    'surface_thermal_radiation_downwards': (0.0, None),
    '2m_temperature_C': (None, None),
    'wind_speed': (0.0, None),
    'surface_pressure': (0.0, None),
    'surface_latent_heat_flux': (None, None),
    'surface_sensible_heat_flux': (None, None),
}

# Variables forced to zero intercept in regression
ZERO_INTERCEPT_VARS = {'WS', 'SW_IN_F_MDS'}


# =============================
# Conversions & basic calc
# =============================

def kelvin_to_celsius ( temp_k ):
    return temp_k - 273.15


def calculate_vpd_and_RH ( t2m_c, dt2m_c ):
    """Magnus–Tetens (Murray, 1967). Returns (VPD[hPa], RH[%])."""
    t2m_arr = np.array(t2m_c)
    dt2m_arr = np.array(dt2m_c)
    a = 6.11 * 10 ** (-2)  # kPa
    mask = t2m_arr < 0
    b = np.where(mask, 21.874, 17.269)
    c = np.where(mask, 265.49, 237.29)
    esat = a * np.exp((b * t2m_arr) / (t2m_arr + c))
    ea = a * np.exp((b * dt2m_arr) / (dt2m_arr + c))
    RH = (ea / esat) * 100.0
    vpd = (esat - ea) * 10.0  # kPa -> hPa
    return vpd, RH


def calculate_wind_speed ( u10, v10 ):
    return np.sqrt(u10 ** 2 + v10 ** 2)


def Pa_to_kPa ( surface_pressure ):
    return surface_pressure / 1000.0


# =============================
# File discovery helpers
# =============================

def _find_first ( path_patterns: List[str] ) -> str:
    for pat in path_patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file found matching patterns: {path_patterns}")


def find_era5land_files ( directory: str ) -> str:
    era5land_files = []
    pattern = re.compile(r'.*ERA5land\.csv$', re.IGNORECASE)
    for root, _, files in os.walk(directory):
        for file in files:
            if pattern.fullmatch(file):
                era5land_files.append(os.path.join(root, file))
    if not era5land_files:
        raise ValueError(f"No *ERA5land.csv file found in {directory}")
    if len(era5land_files) > 1:
        raise ValueError(f"Multiple *ERA5land.csv files found in {directory}: {era5land_files}")
    return era5land_files[0]


def find_fluxnet_files ( file_path: str, source: str ) -> str:
    if source in ("AMF", "FLX"):
        pattern = re.compile(r'.*FLUXNET.*_FULLSET.*\.csv$', re.IGNORECASE)
    elif source == "ICOS":
        pattern = re.compile(r'.*FLUXNET.*\.csv$', re.IGNORECASE)
    else:
        raise ValueError(f"Unknown source: {source}")
    found = []
    for root, _, files in os.walk(file_path):
        for file in files:
            if pattern.fullmatch(file):
                found.append(os.path.join(root, file))
    if not found:
        raise ValueError(f"No FLUXNET file found in {file_path} with source={source}")
    if len(found) > 1:
        raise ValueError(f"Multiple FLUXNET files found in {file_path} with source={source}: {found}")
    return found[0]


# =============================
# Time handling & resampling
# =============================

def fill_missing_values ( df: pd.DataFrame ) -> pd.DataFrame:
    df = df.copy()
    df = df.ffill().bfill()
    if 'RH' in df.columns:
        df['RH'] = df['RH'].interpolate('linear').clip(0, 100)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].interpolate('linear')
    return df


def resample_era5_to_half_hourly ( era5_data ):
    """Resample hourly ERA5-Land data to half-hourly resolution.

    Args:
        era5_data: DataFrame with hourly ERA5-Land data, must contain 'TIMESTAMP_START' column

    Returns:
        DataFrame with half-hourly resolution data
    """
    # Make a copy to avoid modifying original
    era5_data = era5_data.copy()
    # Convert timestamp to datetime if it's not already
    era5_data['TIMESTAMP_END'] = pd.to_datetime(era5_data['TIMESTAMP_END'])
    # Set timestamp as index and sort
    era5_data.set_index('TIMESTAMP_END', inplace=True)
    era5_data.sort_index(inplace=True)
    # Create new half-hourly index
    start_time = era5_data.index.min() - pd.Timedelta(hours=0.5)
    end_time = era5_data.index.max() + pd.Timedelta(hours=1)
    half_hourly_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq='30min'
    )
    # Resample different variable types appropriately
    era5_resampled = pd.DataFrame(index=half_hourly_index)
    # Process each column based on its type
    for col in era5_data.columns:
        if col == 'total_precipitation':
            era5_resampled[col] = era5_data[col].resample('30min').asfreq() / 2
            era5_resampled[col] = era5_resampled[col].interpolate(method='linear').bfill().ffill()
        elif col == 'total_evaporation':
            era5_resampled[col] = era5_data[col].resample('30min').asfreq() / 2
            era5_resampled[col] = era5_resampled[col].interpolate(method='linear').bfill().ffill()
        else:
            era5_resampled[col] = era5_data[col].resample('30min').asfreq()
            era5_resampled[col] = era5_resampled[col].interpolate(method='linear').bfill().ffill()

    # Reset index and add TIMESTAMP_END
    era5_resampled.reset_index(inplace=True)
    era5_resampled.rename(columns={'index': 'TIMESTAMP_END'}, inplace=True)
    era5_resampled['TIMESTAMP_START'] = era5_resampled['TIMESTAMP_END'] - pd.Timedelta(minutes=30)

    # Format timestamps back to original format (YYYYMMDDHHMM)
    era5_resampled['TIMESTAMP_START'] = era5_resampled['TIMESTAMP_START'].dt.strftime('%Y%m%d%H%M')
    era5_resampled['TIMESTAMP_END'] = era5_resampled['TIMESTAMP_END'].dt.strftime('%Y%m%d%H%M')

    return era5_resampled


def filter_data_by_time ( era5_df: pd.DataFrame,
                          flux_df: pd.DataFrame,
                          start_time: pd.Timestamp,
                          end_time: pd.Timestamp ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    era5_df['TIMESTAMP_START'] = _smart_parse_ts(era5_df['TIMESTAMP_START'])
    era5_df['TIMESTAMP_END'] = _smart_parse_ts(era5_df['TIMESTAMP_END'])
    flux_df['TIMESTAMP_START'] = _smart_parse_ts(flux_df['TIMESTAMP_START'])
    flux_df['TIMESTAMP_END'] = _smart_parse_ts(flux_df['TIMESTAMP_END'])
    era5 = era5_df.copy()
    era5['TIMESTAMP_START'] = pd.to_datetime(era5['TIMESTAMP_START'], format='%Y%m%d%H%M')
    era5['TIMESTAMP_END'] = pd.to_datetime(era5['TIMESTAMP_END'], format='%Y%m%d%H%M')

    flux = flux_df.copy()
    flux['TIMESTAMP_START'] = pd.to_datetime(flux['TIMESTAMP_START'], format='%Y%m%d%H%M')
    flux['TIMESTAMP_END'] = pd.to_datetime(flux['TIMESTAMP_END'], format='%Y%m%d%H%M')

    era5_clip = era5[(era5['TIMESTAMP_START'] >= start_time) & (era5['TIMESTAMP_END'] <= end_time)]
    flux_clip = flux[(flux['TIMESTAMP_START'] >= start_time) & (flux['TIMESTAMP_END'] <= end_time)]

    # align on common timestamps
    idx = np.intersect1d(era5_clip['TIMESTAMP_START'].values, flux_clip['TIMESTAMP_START'].values)
    era5_clip = era5_clip[era5_clip['TIMESTAMP_START'].isin(idx)].reset_index(drop=True)
    flux_clip = flux_clip[flux_clip['TIMESTAMP_START'].isin(idx)].reset_index(drop=True)
    return era5_clip, flux_clip


# =============================
# Debiasing core
# =============================

def _ols ( x: np.ndarray, y: np.ndarray, force_zero_intercept: bool ) -> Tuple[float, float, float, int]:
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = x.size
    if n < 2:
        raise ValueError("Not enough overlapping samples to fit OLS.")
    reg = LinearRegression(fit_intercept=not force_zero_intercept)
    reg.fit(x.reshape(-1, 1), y)
    slope = float(reg.coef_[0])
    intercept = 0.0 if force_zero_intercept else float(reg.intercept_)
    r2 = float(reg.score(x.reshape(-1, 1), y))
    return slope, intercept, r2, n


def _mean_var_adjust ( x: np.ndarray, ref: np.ndarray ) -> Tuple[float, float]:
    mx = np.nanmean(x)
    sx = np.nanstd(x)
    mr = np.nanmean(ref)
    sr = np.nanstd(ref)
    s = (sr / sx) if sx > 0 else 1.0
    b = mr - s * mx
    return float(s), float(b)


def _clip_physical ( var: str, arr: np.ndarray ) -> np.ndarray:
    lo, hi = PHYSICAL_BOUNDS.get(var, (None, None))
    out = arr.copy()
    if lo is not None:
        out = np.maximum(out, lo)
    if hi is not None:
        out = np.minimum(out, hi)
    return out


def get_valid_measurement_mask ( flux_df: pd.DataFrame, flux_var: str ) -> np.ndarray:
    qc_var = f"{flux_var}_QC"
    if qc_var in flux_df.columns:
        return (flux_df[qc_var] == 0).values
    return ((flux_df[flux_var] != -9999) & (~flux_df[flux_var].isna())).values


def perform_debiasing ( era5_overlap: pd.DataFrame,
                        flux_overlap: pd.DataFrame,
                        era5_full: pd.DataFrame,
                        var_mapping: Dict[str, str] = None ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Debias by *iterating over all ERA5-Land variables*.

    - If an ERA5 variable has a mapping to a FLUXNET variable (via var_mapping),
      and we have enough valid samples, apply regression-based debiasing (or scaling for P).
    - Otherwise, passthrough the ERA5 variable unchanged and record status.
    """
    if var_mapping is None:
        var_mapping = FLUX_TO_ERA5LAND

    # Build reverse map: era5_var -> flux_var
    era5_to_flux = {era5: flux for flux, era5 in var_mapping.items()}

    # Initialize outputs
    debiased = era5_full[['TIMESTAMP_START', 'TIMESTAMP_END']].copy()
    stats_rows = []
    passthrough_vars: List[str] = []

    # Determine which columns are actual data columns (exclude timestamps)
    time_cols = {'TIMESTAMP_START', 'TIMESTAMP_END'}
    data_cols = [c for c in era5_full.columns if c not in time_cols]
    # Align overlap frames are assumed pre-aligned in filter_data_by_time

    for era5_var in data_cols:
        # If this ERA5 variable has a corresponding flux variable, we attempt debiasing
        flux_var = era5_to_flux.get(era5_var, None)
        # Case A: No mapping provided -> passthrough
        if flux_var is None:
            debiased[era5_var] = era5_full[era5_var]
            passthrough_vars.append(era5_var)
            stats_rows.append({
                'era5_var': era5_var, 'flux_var': None, 'status': 'no_mapping_passthrough',
                'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_train': 0,
                'note': 'ERA5 variable not in mapping; kept as-is.'
            })
            continue

        # Case B: Mapping exists but columns missing -> passthrough
        if (flux_var not in flux_overlap.columns) or (era5_var not in era5_overlap.columns):
            debiased[era5_var] = era5_full.get(era5_var, np.nan)
            passthrough_vars.append(era5_var)
            stats_rows.append({
                'era5_var': era5_var, 'flux_var': flux_var, 'status': 'missing_column',
                'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_train': 0,
                'note': 'passthrough'
            })
            continue

        # Case C: Check valid samples
        valid_mask = get_valid_measurement_mask(flux_overlap, flux_var)
        if valid_mask.sum() < 2:
            debiased[era5_var] = era5_full[era5_var]
            passthrough_vars.append(era5_var)
            stats_rows.append({
                'era5_var': era5_var, 'flux_var': flux_var, 'status': 'insufficient_data',
                'slope': np.nan, 'intercept': np.nan, 'r2': np.nan, 'n_train': int(valid_mask.sum()),
                'note': 'passthrough'
            })
            continue

        # Prepare training arrays
        x_train = era5_overlap.loc[valid_mask, era5_var].to_numpy()
        y_train = flux_overlap.loc[valid_mask, flux_var].to_numpy()

        # Special case: precipitation scaling
        if flux_var == 'P':
            sum_obs = np.nansum(y_train)
            sum_era = np.nansum(x_train)
            scale = (sum_obs / sum_era) if (sum_era > 0) else 1.0
            yhat_full = era5_full[era5_var].to_numpy() * scale
            status = 'debiased_scaling'
            slope, intercept, r2, n_train = scale, 0.0, np.nan, valid_mask.sum()
        else:
            force_zero = (flux_var in ZERO_INTERCEPT_VARS)
            try:
                slope, intercept, r2, n_train = _ols(x_train, y_train, force_zero)
                status = 'debiased_regression'
            except Exception:
                s, b = _mean_var_adjust(x_train, y_train)
                slope, intercept, r2, n_train = s, b, np.nan, valid_mask.sum()
                status = 'fallback_meanvar'
            yhat_full = intercept + slope * era5_full[era5_var].to_numpy()

        # Physical clipping
        yhat_full = _clip_physical(era5_var, yhat_full)
        debiased[era5_var] = yhat_full

        stats_rows.append({
            'era5_var': era5_var, 'flux_var': flux_var, 'status': status,
            'slope': slope, 'intercept': intercept, 'r2': r2, 'n_train': int(n_train),
            'note': ''
        })

    stats_df = pd.DataFrame(stats_rows).set_index('era5_var')
    return debiased, stats_df, passthrough_vars


# =============================
# Evaluation
# =============================

def _metrics ( y_true: np.ndarray, y_pred: np.ndarray ) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if yt.size == 0:
        return {k: np.nan for k in ['RMSE', 'MAE', 'Bias', 'R2', 'NSE']}
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    bias = float(np.mean(yp - yt))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    nse = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'R2': r2, 'NSE': nse}


def evaluate_and_save (
        era5_overlap: pd.DataFrame,
        flux_overlap: pd.DataFrame,
        debiased_full: pd.DataFrame,
        var_mapping: Dict[str, str] ) -> pd.DataFrame:
    """Evaluate only variables that have FLUXNET mapping & columns present.
    Unmapped ERA5 variables are skipped in evaluation but still included in outputs.
    """
    # Build reverse map era5->flux for evaluation
    era5_to_flux = {era5: flux for flux, era5 in var_mapping.items()}

    rows = []
    deb = debiased_full.copy()
    deb['TIMESTAMP_START'] = pd.to_datetime(deb['TIMESTAMP_START'], format='%Y%m%d%H%M')
    overlap_idx = era5_overlap['TIMESTAMP_START'].values
    deb = deb[deb['TIMESTAMP_START'].isin(overlap_idx)].reset_index(drop=True)

    for era5_var, flux_var in era5_to_flux.items():
        if (flux_var not in flux_overlap.columns) or (era5_var not in era5_overlap.columns) or (
                era5_var not in deb.columns):
            continue
        valid = get_valid_measurement_mask(flux_overlap, flux_var)
        y_true = flux_overlap.loc[valid, flux_var].to_numpy()
        y_era5 = era5_overlap.loc[valid, era5_var].to_numpy()
        y_deb = deb.loc[valid, era5_var].to_numpy()
        base = _metrics(y_true, y_era5)
        debm = _metrics(y_true, y_deb)
        rows.append({
            'variable': era5_var,
            'flux_var': flux_var,
            **{f'ERA5_{k}': v for k, v in base.items()},
            **{f'DEBIASED_{k}': v for k, v in debm.items()}
        })
    eval_df = pd.DataFrame(rows)
    return eval_df


def _smart_parse_ts(series: pd.Series) -> pd.Series:
    """
    兼容解析：
    - 12位纯数字 YYYYMMDDHHMM（常见于 FLUXNET）
    - ISO/混合格式（如 1990-12-31 19:00:00）
    - 已是 datetime 则原样返回
    """
    if np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series, errors='coerce')

    s = series.astype(str)
    is12 = s.str.fullmatch(r"\d{12}").fillna(False)
    # 若绝大多数是12位数字，用固定format更稳
    if is12.mean() > 0.8:
        return pd.to_datetime(s, format="%Y%m%d%H%M", errors="coerce")

    # 否则用通用解析（能吃 ISO/混合）
    return pd.to_datetime(s, errors="coerce")


# =============================
# IO helpers
# =============================

def debias_era5land_HH ( site_path,era_path, site_id, site_source, start_time, end_time, debias_file_path):
    var_mapping = FLUX_TO_ERA5LAND
    era5land_data_path = find_era5land_files(era_path)
    flux_data_path = find_fluxnet_files(site_path, site_source)
    print(f'ERA5-Land file: {era5land_data_path} found')
    print(f'FLUXNET file: {flux_data_path} found')
    era5land_hourly = pd.read_csv(era5land_data_path)
    flux_data = pd.read_csv(flux_data_path)
    hour_file_name = f'{site_id}_Era5land_HH.csv'
    resample_file_path = os.path.join(site_path, hour_file_name)
    # resample to half-hourly and save
    era5land_resample = resample_era5_to_half_hourly(era5land_hourly)
    era5land_resample.to_csv(resample_file_path, index=False)
    print(f'Resample(HH) file saved to {resample_file_path}')
    assert not era5land_resample.isnull().any().any(), "There are missing values after resampling"
    # filter overlap window
    start_time = pd.to_datetime(start_time, format='%Y%m%d%H%M')
    end_time = pd.to_datetime(end_time, format='%Y%m%d%H%M')
    era5_overlap, flux_overlap = filter_data_by_time(era5land_resample, flux_data, start_time, end_time)
    assert len(era5_overlap) == len(flux_overlap), "ERA5 and FLUXNET lengths mismatch after filtering"
    # debias and save
    debiased_df, stats_df, passthrough = perform_debiasing(era5_overlap, flux_overlap, era5land_resample, var_mapping)
    debiased_file_name = f'{site_id}_Era5land_HH_debias.csv'
    try:
        debiased_file_path = os.path.join(debias_file_path, debiased_file_name)
        debiased_df.to_csv(debiased_file_path, index=False)
        print(f"[OK] Saved debias file: {debiased_file_path}")
    except Exception as e:
        print(f"[ERROR] {site_id}: failed to save debias file: {e}")
    eval_df = evaluate_and_save(era5_overlap, flux_overlap, debiased_df, var_mapping)
    return stats_df, eval_df


def debias_era5land_HR ( site_path, era_path,site_id, site_source, start_time, end_time, debias_file_path):
    var_mapping = FLUX_TO_ERA5LAND
    era5land_data_path = find_era5land_files(era_path)
    flux_data_path = find_fluxnet_files(site_path, site_source)
    print(f'ERA5-Land file: {era5land_data_path} found')
    print(f'FLUXNET file: {flux_data_path} found')
    era5land_hr = pd.read_csv(era5land_data_path)
    era5land_hr = fill_missing_values(era5land_hr)
    hr_file_path = os.path.join(site_path, f'{site_id}_Era5land_HR.csv')
    assert not era5land_hr.isnull().any().any(), "There are missing values after filling"
    era5land_hr.to_csv(hr_file_path, index=False)
    print(f'Resample(HR) file saved to {hr_file_path}')
    # filter overlap
    start_time = pd.to_datetime(start_time, format='%Y%m%d%H%M')
    end_time = pd.to_datetime(end_time, format='%Y%m%d%H%M')
    era5_overlap, flux_overlap = filter_data_by_time(era5land_hr, pd.read_csv(flux_data_path), start_time, end_time)
    # debias on HR timeline directly
    debiased_df, stats_df, passthrough = perform_debiasing(era5_overlap, flux_overlap, era5land_hr, var_mapping)
    debiased_file_name = f'{site_id}_Era5land_HR_debias.csv'
    try:
        debiased_file_path = os.path.join(debias_file_path, debiased_file_name)
        debiased_df.to_csv(debiased_file_path, index=False)
        print(f"[OK] Saved debias file: {debiased_file_path}")
    except Exception as e:
        print(f"[ERROR] {site_id}: failed to save debias file: {e}")
    # evaluate
    eval_df = evaluate_and_save( era5_overlap, flux_overlap, debiased_df, var_mapping)
    return stats_df, eval_df
