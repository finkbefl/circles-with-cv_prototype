# Functions for operations with csv files

# Import the external packages
# Operating system functionalities
import os
# To handle pandas data frames
import pandas as pd


def load_data(dirname, filename, sep=',', utf16=False):
    """
    Loading the data from csv file to DataFrame
    ----------
    Parameters:
        dirname : String
            The name of the directory
        filename : String
            The name of the file
        sep : String
            The seperator (default is ',')
        utf16 : boolean
            If the file should be loaded with utf-16 encoding (default is False)
    ----------
    Returns:
        The data as pandas DataFrame
    """

    # Join the filepath of the raw data file
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "data", dirname, filename)

    # Read the data from CSV file
    if utf16 :
        # With utf-16 encoding
        with open(filepath,encoding='UTF-16') as f:
            data_raw = pd.read_csv(f, sep=sep)
    else :
        data_raw = pd.read_csv(filepath, sep=sep)

    return data_raw

#########################################################

def save_data(df, dirname, filename):
    """
    Save the data into a csv file
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
        dirname : String
            The name of the directory
        filename : String
            The name of the file
    ----------
    Returns:
        no return
    """

    # Join the filepath of the raw data file and create directory if it not exist
    file_dir = os.path.join(os.path.dirname(__file__), "..", "..", dirname)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    filepath = os.path.join(file_dir, filename)

    # Save the data to CSV file
    df.to_csv(filepath, index=False)

#########################################################

def convert_series_into_date(df_series, unit=None):
    """
    Function to convert the date objects of a pandas series into DateTime
    ----------
    Parameters:
        df : pandas.core.series.Series
            The data
        unit : the unit of the date objects
    ----------
    Returns:
        The converted data as series
    """

    # Convert the date objects into DateTime (raise an exception when parsing is invalid)
    return pd.to_datetime(df_series, errors='raise', utc=True, unit=unit)