#############################################################
# Author: Vadim Litvinov
# Date: 22 July 2024
#############################################################
import os
import re
import time
import pandas as pd
from typing import Union


def loadAppendChannelsXlTable(filename: str, directory: str, libraries: bool, lib_shared_genes: list[str],
                              dataframes: list[pd.DataFrame]) -> None:
    """
    Given a file name, this function checks if it is readout to gene to job Excel file. If yes, it checks if we have
    library number or not (par^-FISH or not), and appends them to a list
    :param filename: file in the directory
    :param directory: path to directory
    :param libraries: determining if there are multiple libraries in the experiment data or not
    :param lib_shared_genes: list of shared genes between libraries
    :param dataframes: list of dataframes, each dataframe is a table loaded from a file
    :return: None
    """
    if filename.endswith('.xlsx') and 'readout' in filename.lower():
        filepath = os.path.join(directory, filename)
        df = pd.read_excel(filepath)
        # Now create additional column of the lib name
        if libraries:
            df['lib_num'] = re.search(r'_(.*?)\.', filename).group(1)
        elif not libraries and not lib_shared_genes:
            df['lib_num'] = 'constant'
        dataframes.append(df)


def readXlsFile(directory: str, lib_shared_genes: list[str], libraries: bool) -> pd.DataFrame:
    """
    Read the cell by gene files and concatenate them
    :param directory: where the files are lying
    :param lib_shared_genes: list of shared genes between libraries
    :param libraries: determining if there are multiple libraries in the experiment data or not
    :return: dataframe of all the cell by gene files
    """
    start_time = time.time()
    dataframes = []
    for filename in os.listdir(directory):
        loadAppendChannelsXlTable(filename, directory, libraries, lib_shared_genes, dataframes)
    big_dataframe = pd.concat(dataframes, ignore_index=True)
    end_time = time.time()
    print('Loading genes list runtime:', end_time - start_time, 'seconds')
    return big_dataframe


def updateColumnName(col: str, shared_genes: list[str], suffix: str) -> str:
    """Adds suffix to all genes that are shared between libraries"""
    if col in shared_genes:
        return f"{col}__{suffix}"
    else:
        return col


def loadAppendCellByGeneTxtTable(filename: str, directory: str, control_genes_etc: list[str], control_per_library: bool,
                                 dataframes: list[pd.DataFrame]) -> None:
    """
    Given filename, this function checks if it is a cell by gene .txt file, loads it, and concatenates them.
    Additionally, adds the library number as the suffix
    :param filename:
    :param directory:
    :param control_genes_etc:
    :param control_per_library:
    :param dataframes:
    :return:
    """
    if filename.endswith('.txt'):
        # Check if the file name contains "cell", "gene" and "norm"
        if 'cell' in filename.lower() and 'gene' in filename.lower() and 'norm' in filename.lower():
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, delimiter='\t'
                             # , nrows=100
                             )
            if control_genes_etc and control_per_library:
                lib = filename.split('_')[1]
                df.columns = [updateColumnName(col, control_genes_etc, lib) for col in df.columns]
            dataframes.append(df)


def readTxtFile(directory: str, control_genes_etc: list[str], control_per_library: bool) -> pd.DataFrame:
    """
    Read the text files in the folder and concatenate them
    :param directory: where the files are lying
    :param control_genes_etc: list of special genes for control etc.
    :param control_per_library: control genes per library or one general
    :return: dataframe of all the text files
    """
    start_time = time.time()
    dataframes = []
    for filename in os.listdir(directory):
        loadAppendCellByGeneTxtTable(filename, directory, control_genes_etc, control_per_library, dataframes)
    big_dataframe = pd.concat(dataframes, ignore_index=True)
    end_time = time.time()
    print('Loading cell by gene runtime:', end_time - start_time, 'seconds')
    return big_dataframe


def readCsvFile(file_path: str) -> Union[pd.DataFrame, None]:
    """Reads statistics summary csv file"""
    try:
        df = pd.read_csv(file_path,
                         index_col='gene_od',
                         #nrows=100
                         )
        print('Statistics summary loaded.')
        return df
    except Exception as exception:
        print(f"Error reading the file: {exception}")
        return None


def writeToFile(data: pd.DataFrame, name: str) -> None:
    """Saves dataframe to file"""
    data.to_csv(name, index=True)


def writeResToFile(data: pd.DataFrame, name: str) -> None:
    """Saves residuals dataframe to file"""
    data.to_csv(name, index=True, columns=data.columns, header=data.columns)
