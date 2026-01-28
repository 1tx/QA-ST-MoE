import pandas as pd
import re
import os


def find_fluxnet_files ( file_path, source ) -> str:
    """
    Find FLUXNET data files in the specified directory

    Parameters:
        file_path (str): The root directory path to search
        source (str, optional): The type of data source, optional "AMF", "FLX" or "ICOS"

    Returns:
        str: The path of the found FLUXNET file

    Exceptions:
        ValueError: If multiple matching files are found or no file is found

    FLUXNET file matching rules:
    - If source is "AMF" or "FLX": The filename must match "FLUXNET.*_FULLSET.*\.csv$"
    - If source is "ICOS": The filename only needs to contain "FLUXNET" and end with .csv
    - If source is not specified: Use the general matching rule (contains "FLUXNET" and ends with .csv)
    """
    fluxnet_files = []
    # Determine the matching pattern based on the data source
    if source in ("AMF", "FLX"):
        pattern = re.compile(r'.*FLUXNET.*_FULLSET.*\.csv$', re.IGNORECASE)
    elif source == "ICOS":
        pattern = re.compile(r'.*FLUXNET.*\.csv$', re.IGNORECASE)
    else:
        print('The data source does not exist currently')
        pattern = re.compile(r'.*FLUXNET.*\.csv$', re.IGNORECASE)

    for root, _, files in os.walk(file_path):
        for file in files:
            if pattern.fullmatch(file):
                file_path = os.path.join(root, file)
                fluxnet_files.append(file_path)

    # Check the number of results
    if not fluxnet_files:
        raise ValueError(f"No FLUXNET file found in {file_path} with source={source}")
    if len(fluxnet_files) > 1:
        raise ValueError(f"Multiple FLUXNET files found in {file_path} with source={source}: {fluxnet_files}")
    return fluxnet_files[0]


def load_site_information ( site_information_path ):
    # If the filling flag is valid, check if the site information file exists
    if os.path.exists(site_information_path):
        # If it does, read the file using pandas
        site_information = pd.read_excel(site_information_path, sheet_name='site_info', engine='openpyxl')
    else:
        # If it doesn't, raise a ValueError
        raise ValueError(f"Site information file {site_information_path} does not exist")
    # Return the site information
    return site_information


def filter_sites ( site_information, filling_var, max_missing_percentage=50 ):
    """
    Filter sites based on missing data statistics, returning sites with a missing rate below a specified threshold for the specified variable.

    Parameters:
        site_information (list): A list containing site information, where each element is a dictionary containing the site's name and other information.
        filling_var (str): The name of the variable to check for missing data.
        max_missing_percentage (float): The maximum allowed missing rate, default is 50%.

    Returns:
        list: A list of site information that meets the conditions.
    """
    # Convert site_information to DataFrame for operation
    site_info_df = pd.DataFrame(site_information)
    # Load the corresponding statistical file based on filling_flag
    stat_data_file_path = './data/config/statistics/missing_statistics.xlsx'
    if not os.path.exists(stat_data_file_path):
        raise FileNotFoundError(f"Statistical file {stat_data_file_path} does not exist")
    stat_data = pd.read_excel(stat_data_file_path)
    # Filter out sites with a missing rate below max_missing_percentage for the specified variable
    acceptable_sites = stat_data[(stat_data['VARIABLE'] == filling_var) &
                                 (stat_data['missing_percentage'] < max_missing_percentage)]
    # Get the site IDs and data sources that meet the conditions
    acceptable_site_ids = acceptable_sites[['SITE_ID', 'SOURCE']].drop_duplicates()
    # Filter site_information, retaining only sites that meet the conditions
    filtered_sites = site_info_df.merge(acceptable_site_ids, on=['SITE_ID', 'SOURCE'], how='inner')
    return filtered_sites  # Return the filtered site information list
